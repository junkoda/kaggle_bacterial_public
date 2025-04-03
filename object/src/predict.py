"""
Compute Local CV
"""
import numpy as np
import time
import yaml
import signal
import argparse

from sklearn.model_selection import KFold
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn

# Modules in this directory
from data_pred import Dataset, create_df, preprocess
from model import Model
from metric import load_label, compute_score

signal.signal(signal.SIGINT, signal.SIG_DFL)  # stop with Ctrl-C


def predict(tomo_path: Path,
            models: list[nn.Module],
            cfg: dict) -> dict:
    """
    Predict moter coordinate for one tomo_id

    Args:
      models (list[nn.Module]): Pytorch models
      dataset (Dataset): for one tomo_id
    """
    assert len(models) > 0
    tomo_id = tomo_path.name
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    use_amp = cfg['amp']

    dataset = Dataset(tomo_path)
    loader = dataset.loader(batch_size, num_workers)

    device = next(models[0].parameters()).device

    # Loop over all slices in one tomo_id
    best = (0, None)
    for d in loader:
        img = d['img'].to(device)  # input image (batch_size, 1, H, W)
        img = preprocess(img)

        y_pred_sum, t_pred_sum = None, None
        for model in models:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda',
                                        enabled=use_amp,
                                        dtype=torch.float16):
                    y_pred, t_pred = model(img)

            y_pred = y_pred.sigmoid()         # (batch_size, 1, h, w)

            if y_pred_sum is None:
                y_pred_sum = y_pred
                t_pred_sum = t_pred
            else:
                y_pred_sum += y_pred
                t_pred_sum += t_pred

        y_pred_max = y_pred_sum.max().item() / len(models)

        # Keep most probable coordinate
        if y_pred_max > best[0]:
            bs, _, h, w = y_pred_sum.shape

            argmax = torch.unravel_index(y_pred.argmax(), y_pred_sum.shape)  # b, ch, iy, ix
            i, _, iy, ix = [t.item() for t in argmax]
            slice_number = d['slice_number'][i].item()
            offset = t_pred_sum[i, :, iy, ix].cpu().numpy() / len(models)  # (2, )

            # Compute coodinate in original pixels
            H, W = d['shape'][i].numpy()    # Original image size
            x = (ix + offset[0]) * (W / w)
            y = (iy + offset[1]) * (H / h)

            best = (y_pred_max, slice_number, y, x)

    assert best[1] is not None

    # Return prediction
    n_slices = len(dataset.filenames)
    pred = {'tomo_id': tomo_id,
            'n_slices': n_slices,
            'y_pred': best[0],
            'zyx': best[1:]}
    return pred


#
# Main
#
device = torch.device('cuda')

# Command-line options
parser = argparse.ArgumentParser()
parser.add_argument('filename', help='devel.yml')
parser.add_argument('--data-type', default='test', help='train or test')
arg = parser.parse_args()

tb = time.time()

# Config yaml
name = Path(arg.filename).name.replace('.yml', '')
with open(arg.filename, 'r') as f:
    cfg = yaml.safe_load(f)

input_path = Path(cfg['data']['input_path'])
model_path = Path('experiments') / name
use_amp = cfg['train']['amp']
th = 0.5

# Data
df = create_df(input_path)
labeld = load_label(input_path)

# KFold
nfold = cfg['kfold']['k']
folds = cfg['kfold']['folds']  # list[int]
kfold = KFold(n_splits=nfold, random_state=42, shuffle=True)

scores = []
for ifold, (idx_train, idx_val) in enumerate(kfold.split(df)):
    if ifold not in folds:
        continue

    df_val = df.iloc[idx_val]
    print('Fold %d / %d data %d' % (ifold, nfold, len(df_val)))

    # Model
    model = Model(cfg['model'], pretrained=False, verbose=False)
    model_filename = '%s/model%d.pytorch' % (model_path, ifold)
    model.load_state_dict(torch.load(model_filename, weights_only=True))
    model.to(device)
    model.eval()

    # Predict
    cfg_pred = {
        'batch_size': 16,
        'num_workers': 4,
        'amp': use_amp,
    }

    preds = []
    for i, r in tqdm(df_val.iterrows(), total=len(df_val), desc='Predict'):
        tomo_path = Path(r['tomo_dir'])
        pred = predict(tomo_path, [model, ], cfg_pred)
        preds.append(pred)

    # Score
    sc = compute_score(preds, labeld, th)
    print('Score %.6f precision %.4f recall %.4f' % (
            sc['score'], sc['precision'], sc['recall']))
    scores.append(sc['score'])

    del model

dt = time.time() - tb
print('Time %.2f sec' % dt)

if len(scores) > 1:
    print(' '.join(['%.4f' % s for s in scores]))
    print('%.4f ± %.4f' % (np.mean(scores), np.std(scores, ddof=1)))

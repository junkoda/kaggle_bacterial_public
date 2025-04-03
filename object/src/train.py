import os
import time
import yaml
import shutil
import signal
import argparse

from sklearn.model_selection import KFold
from tqdm import tqdm
from pathlib import Path

import torch

# Modules in this directory
from data import Data
from model import Model
from lr_scheduler import Scheduler

signal.signal(signal.SIGINT, signal.SIG_DFL)  # stop with Ctrl-C
device = torch.device('cuda')


# Command-line options
parser = argparse.ArgumentParser()
parser.add_argument('filename', help='devel.yml')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--debug', action='store_true')
arg = parser.parse_args()

debug = arg.debug
tb_global = time.time()

# Config yaml
name = 'debug' if arg.debug else \
        os.path.basename(arg.filename).replace('.yml', '')
with open(arg.filename, 'r') as f:
    cfg = yaml.safe_load(f)

input_path = Path(cfg['data']['input_path'])
data_path = Path(cfg['data']['data_path'])

# Output directory
odir = Path('experiments')
if not odir.exists():
    raise FileNotFoundError('Output directory `%s` does not exist.' % odir)

odir = odir / name
if odir.exists():
    if (not arg.overwrite) and (not debug):
        model_files = list(odir.glob('*.pytorch'))
        if model_files:
            raise FileExistsError('Model already exists in `%s`.' % odir)
else:
    odir.mkdir()
    print(odir, 'created')


# Copy files
shutil.copy(arg.filename, odir / 'config.yml')
src = Path('train')
for file in ['train.py', 'data.py', 'model.py', ]:
    shutil.copy(src / file, odir)

# Data
data = Data(cfg, debug=debug)
print(data)

# KFold
nfold = cfg['kfold']['k']
folds = cfg['kfold']['folds']  # list[int]
kfold = KFold(n_splits=nfold, random_state=42, shuffle=True)

# Hyperparameters
weight_decay = float(cfg['train']['weight_decay'])
val_frequency = cfg['validate']['frequency']

use_amp = cfg['train']['amp']
assert isinstance(use_amp, bool)
print('use_amp:', use_amp)


#
# Evaluate
#
def evaluate(model, loader_val):
    """
    Compute validation loss
    TODO: Compute fbeta score
    """
    tb = time.time()

    was_training = model.training
    model.eval()

    correct = 0
    loss_sum, loss_seg_sum, loss_reg_sum, n_sum = 0, 0, 0, 0
    for d in loader_val:
        img = d['img'].to(device)        # (batch_size, 1, H, W)
        mask = d['mask'].to(device)      # (batch_size, 1, h, w)
        offset = d['offset'].to(device)  # (batch_size, 2, h, w)
        batch_size = len(img)

        # Predict
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda',
                                    enabled=use_amp,
                                    dtype=torch.float16):
                y_pred, t_pred = model(img)
                loss_seg, loss_reg, loss = model.loss(y_pred, t_pred, mask, offset)

        # Loss
        n_sum += batch_size
        loss_sum += batch_size * loss.item()
        loss_seg_sum += batch_size * loss_seg.item()
        loss_reg_sum += batch_size * loss_reg.item()

        # Maximum grid accuracy
        correct += torch.sum(y_pred.view(batch_size, -1).argmax(dim=1) ==
                             mask.view(batch_size, -1).argmax(dim=1)).item()

    model.train(was_training)

    dt = time.time() - tb
    ret = {'loss': loss_sum / n_sum,
           'loss_seg': loss_seg_sum / n_sum,
           'loss_reg': loss_reg_sum / n_sum,
           'accuracy': correct / n_sum,
           'dt': dt}
    return ret


#
# Train
#
for ifold, (idx_train, idx_val) in enumerate(kfold.split(data.df)):
    if ifold not in folds:
        continue

    print('Fold %d / %d' % (ifold, nfold))

    # Data
    loader_train = data.loader(idx_train, cfg['loader'], shuffle=True, augment=True)
    loader_val = data.loader(idx_val, cfg['loader'])
    nbatch = len(loader_train)
    print('Train %d batches x %d' % (nbatch, cfg['loader']['batch_size']))

    # Model
    model = Model(cfg['model'])
    model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  weight_decay=weight_decay)
    scheduler = Scheduler(optimizer, cfg['scheduler'])
    epochs = 2 if debug else len(scheduler)
    print('%d epochs' % epochs)

    scaler = torch.GradScaler('cuda', enabled=use_amp)

    # Training loop
    tb = time.time()
    dt_val = 0
    loss_sum, loss_seg_sum, loss_reg_sum, n_sum = 0, 0, 0, 0
    print('Epoch  loss                              acc    lr        time')
    for iepoch in range(epochs):
        val_steps = [i * (nbatch // val_frequency) - 1 for i in range(1, val_frequency)] + [nbatch - 1, ]

        disable = nbatch < 1000
        for ibatch, d in enumerate(tqdm(loader_train, disable=disable)):
            # Data
            img = d['img'].to(device)        # (batch_size, 1, H, W)
            mask = d['mask'].to(device)      # (batch_size, 1, h, w)
            offset = d['offset'].to(device)  # (batch_size, 2, h, w)
            batch_size = len(img)

            optimizer.zero_grad()

            # Predict
            with torch.autocast(device_type='cuda',
                                enabled=use_amp,
                                dtype=torch.float16):
                y_pred, t_pred = model(img)
                loss_seg, loss_reg, loss = model.loss(y_pred, t_pred, mask, offset)

            # Record training loss
            n_sum += batch_size
            loss_sum += batch_size * loss.item()
            loss_seg_sum += batch_size * loss_seg.item()
            loss_reg_sum += batch_size * loss_reg.item()

            # Backpropagate
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            ep = iepoch + (ibatch + 1) / nbatch
            scheduler.step(ep)

            if ibatch in val_steps:
                val = evaluate(model, loader_val)
                dt_val += val['dt']

                loss_train = loss_sum / n_sum
                loss_seg = loss_seg_sum / n_sum
                loss_reg = loss_reg_sum / n_sum
                lr = optimizer.param_groups[0]['lr']

                dt = time.time() - tb
                tqdm.write('%5.2f %7.4f %7.4f  %7.4f %7.4f  %6.3f  %5.1e  %5.2f %5.2f min' % (ep,
                           loss_train, val['loss'],
                           loss_seg, loss_reg, val['accuracy'],
                           lr, dt_val / 60, dt / 60))

                loss_sum, loss_seg_sum, loss_reg_sum, n_sum = 0, 0, 0, 0

    # Save model
    model.to('cpu')
    model.eval()
    ofilename = '%s/model%d.pytorch' % (odir, ifold)
    torch.save(model.state_dict(), ofilename)
    print(ofilename, 'written')


dt = time.time() - tb_global
print('Total time %.2f min' % (dt / 60))

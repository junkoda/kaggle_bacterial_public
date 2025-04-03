"""
Compute fbeta score

$ python3 metric.py submission.csv
"""
import numpy as np
import pandas as pd
import yaml
import argparse
from pathlib import Path


def fbeta(tp, num_pred, num_true):
    beta = 2

    precision = tp / num_pred if num_pred > 0 else 0
    recall = tp / num_true if num_true > 0 else 0

    num = (1 + beta ** 2) * precision * recall
    denom = (beta ** 2 * precision) + recall
    score = num / denom if denom > 0 else 0

    return score, precision, recall


def load_label(input_path):
    df = pd.read_csv(input_path / 'train_labels.csv')

    labeld = {}  # dict: tomo_id -> label (dict)
    for tomo_id, df1 in df.groupby('tomo_id'):
        r = df1.iloc[0]
        m = r.iloc[9]    # number of motors

        if m > 0:
            zyx = df1.iloc[:, 2:5].to_numpy()
        else:
            zyx = np.zeros((0, 3))

        shape = tuple(r.iloc[5:8])
        d = {'shape': shape,
             'spacing': r.iloc[8],
             'zyx': zyx,  # array (m, 3)
        }
        labeld[tomo_id] = d

    return labeld


def compute_score(preds: list[dict],
                  labeld: dict,
                  th=0):
    eps = 1000  # distance threshold (angstrom) in the competition metric

    tp, num_pred, num_true = 0, 0, 0
    for pred in preds:
        tomo_id = pred['tomo_id']
        label = labeld[tomo_id]

        if pred['y_pred'] < th:
            continue

        num_pred += 1

        zyx_true = label['zyx']
        if len(zyx_true) == 0:
            continue

        num_true += 1

        zyx_pred = np.array(pred['zyx']).reshape(1, 3)
        r2 = np.sum((zyx_pred - zyx_true) ** 2, axis=1).min()
        r = label['spacing'] * np.sqrt(r2)

        if r <= eps:
            tp += 1

    # Score statistics
    score, precision, recall = fbeta(tp, num_pred, num_true)

    ret = {'score': score,
           'precision': precision,
           'recall': recall,
           'count': (tp, num_pred, num_true)}
    return ret


def preds_from_submission(filename):
    submit = pd.read_csv(filename)

    preds = []
    for i, r in submit.iterrows():
        tomo_id = r['tomo_id']
        zyx = (r['Motor axis 0'], r['Motor axis 1'], r['Motor axis 2'])

        y_pred = -1 if zyx[0] == -1 else 0

        pred = {'tomo_id': tomo_id,
                'y_pred': y_pred,
                'zyx': zyx}
        preds.append(pred)

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='baseline.yml')
    parser.add_argument('--submit', default='submission.csv')
    arg = parser.parse_args()

    # Input path
    with open(arg.filename, 'r') as f:
        cfg = yaml.safe_load(f)
    input_path = Path(cfg['data']['input_path'])

    preds = preds_from_submission(arg.submit)
    labeld = load_label(input_path)
    sc = compute_score(preds, labeld)

    print('Score %.6f' % sc['score'])
    print('Precision / recall %.4f %.4f' % (sc['precision'], sc['recall']))
    print(sc['count'])


if __name__ == '__main__':
    main()

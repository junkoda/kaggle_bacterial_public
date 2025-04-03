"""
Prepare training data

* Use only positive slices around motors
* Resize images to 640x640
* Label coordinates are rescaled to pixels in 640x640
"""
import numpy as np
import polars as pl
import yaml
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path


def load_labels(input_path: Path):
    """
    Create dataframe for KFold split and label dict

    * df (pd.DataFrame): Mainly tomo_id
    * labeld (dict): tomo_id -> coordinates zyx (m, 3)
    """
    label_path = input_path / 'train_labels.csv'
    if not label_path.exists():
        raise FileNotFoundError(str(label_path))

    # Load label
    df = pl.read_csv(label_path)

    # Positive only
    df = df.filter(pl.col('Number of motors') > 0)

    rows = []
    labeld = {}  # dict: tomo_id -> label (dict)
    for grp, df1 in df.group_by('tomo_id', maintain_order=True):
        tomo_id = grp[0]
        r = df1.row(0)
        m = r[9]    # number of motors

        row = {'tomo_id': tomo_id,
               'm': m,
               'nz': r[5],
               'ny': r[6],
               'nx': r[7],
               'spacing': r[8],
        }
        rows.append(row)

        if m > 0:
            zyx = df1[:, 2:5].to_numpy()
        else:
            zyx = np.zeros((0, 3))

        shape = tuple(r[5:8])
        d = {'shape': shape,
             'spacing': r[8],
             'zyx': zyx,  # array (m, 3)
        }
        labeld[tomo_id] = d

    df = pl.DataFrame(rows)

    return df, labeld


def normalize_slice(img: np.ndarray):
    q_min, q_max = np.quantile(img, [0.02, 0.98])
    img = np.clip(img, q_min, q_max)
    img = 255 * (img - q_min) / (q_max - q_min)

    return img


def create_data(df: pl.DataFrame,
                labeld: dict,
                input_path: str,
                output_path: str):
    """
    Args
      df (pl.DataFrame): tomo_id
      labeld (dict):     tomo_id -> zyx array (m, 3)
    """
    dz = 4
    size = (640, 640)

    # Loop over volumes (tomo_ids)
    count = 0
    desc = 'Prepare'
    for r in tqdm(df.iter_rows(named=True), total=len(df), desc=desc):
        tomo_id = r['tomo_id']
        n_slices = r['nz']
        d = labeld[tomo_id]
        zyx = d['zyx']  # array (m, 3)

        # Loop over positive slices
        for z, y, x in zyx:
            iz = int(z)
            z_begin = max(0, iz - dz)
            z_end = min(iz + dz + 1, n_slices)

            for iz in range(z_begin, z_end):
                filename = '%s/train/%s/slice_%04d.jpg' % (input_path, tomo_id, iz)

                # Write image (no resize?)
                ofilename = '%s/images/%s_%04d.jpg' % (output_path, tomo_id, iz)
                img = Image.open(filename)
                W, H = img.size

                img = img.resize(size, Image.BILINEAR)
                img = np.array(img)
                img = normalize_slice(img)
                img = img.astype(np.uint8)

                Image.fromarray(img).save(ofilename)

                cx = (size[0] / W) * x
                cy = (size[1] / H) * y

                # Write labels; may overwrite if multiple moters in volume
                ofilename = '%s/labels/%s_%04d.txt' % (output_path, tomo_id, iz)
                with open(ofilename, 'w') as f:
                    f.write('%f %f\n' % (cx, cy))

                count += 1

    print('%s written / %d tomo_ids, %d slices' % (output_path, len(df), count))


def main():
    # Command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    arg = parser.parse_args()

    with open(arg.filename, 'r') as f:
        cfg = yaml.safe_load(f)

    input_path = Path(cfg['data']['input_path'])
    data_path = Path(cfg['data']['data_path'])

    # Output directory
    if not data_path.exists():
        raise FileNotFoundError('Output directory `%s` does not exist.' % str(data_path))

    for sub in ['images', 'labels']:
        sub_path = data_path / sub
        sub_path.mkdir(exist_ok=True)

    # Load dataframe
    df, labeld = load_labels(input_path)

    create_data(df, labeld, input_path, data_path)


if __name__ == '__main__':
    main()

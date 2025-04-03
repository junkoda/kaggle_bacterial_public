import numpy as np
import pandas as pd
from pathlib import Path

import torch
import albumentations as A

from torch.utils.data import DataLoader
from PIL import Image


def create_augmentation(augment):
    if not augment:
        return None

    aug = A.Compose([
            # Add more augmentations ...
            A.D4(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    return aug


def create_df(data_path: Path):
    """
    Create DataFrame of tomo_ids for KFold split

    Returns:
      df (pd.DataFrame): tomo_ids
      datad (dict): tomo_id -> list of slice numbers
    """
    filenames = sorted(data_path.glob('images/*.jpg'))  # list[Path]
    assert filenames

    datad = {}
    for filename in filenames:
        basename = filename.name
        tomo_id = basename[:11]
        slice_number = basename[12:16]
        assert basename[16:] == '.jpg'

        if tomo_id not in datad:
            datad[tomo_id] = {'slice_numbers': list()}

        d = datad[tomo_id]
        d['slice_numbers'].append(slice_number)

    tomo_ids = list(datad.keys())
    tomo_ids.sort()

    df = pd.DataFrame({'tomo_id': tomo_ids})

    return df, datad


def create_slices(df, datad):
    """
    Create list of slices
    One slice is one datum in Dataset
    """
    slices = []
    for tomo_id in df['tomo_id']:
        d = datad[tomo_id]
        for num in d['slice_numbers']:
            slices.append((tomo_id, num))

    return slices


def create_mask(xy, shape):
    """
    xy (array):    coordinate in (H, W)
    shape (tuple): image size (H, W)

    Retruns:
      mask (array):   (1, h, w)
      offset (array): (2, h, w)

    h, w = H / 32, W / 32
    """
    assert xy.shape == (1, 2)
    fac = 32  # corasening factor

    H, W = shape
    h, w = H // fac, W // fac

    mask = np.zeros((1, h, w), dtype=int)
    offset = np.zeros((2, h, w), dtype=np.float32)

    for xy1 in xy:
        x, y = xy1 / fac
        ix, iy = int(x), int(y)
        if (0 <= ix < w) and (0 <= iy < h):
            mask[0, iy, ix] = 1
            offset[0, iy, ix] = x - ix
            offset[1, iy, ix] = y - iy

    return mask, offset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, slices, data, *, augment=False):
        self.slices = slices
        self.data = data
        self.aug = create_augmentation(augment)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        tomo_id, slice_number = self.slices[i]

        # Image
        img = self.data.load_image(tomo_id, slice_number)
        H, W = img.shape

        # Label
        img = np.expand_dims(img, axis=2)
        a = self.data.load_label(tomo_id, slice_number)  # (1, 2)

        if self.aug is not None:
            transformed = self.aug(image=img, keypoints=a)
            img = transformed['image']
            a = transformed['keypoints']

        mask, offset = create_mask(a, img.shape[:2])

        img = img.reshape(1, H, W)           # (1, size, size)

        ret = {'img': img,
               'mask': mask,      # (1, size / 32, size / 32)
               'offset': offset,
        }

        return ret


class Data:
    def __init__(self, cfg, *, debug=False):
        self.data_path = Path(cfg['data']['data_path'])
        self.df, self.datad = create_df(self.data_path)

        if debug:
            self.df = self.df.iloc[:10]

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        s = 'Data(%d)' % len(self)
        return s

    def load_image(self, tomo_id, slice_number):
        """
        Returns
          img (array[float32]): (size, size)
        """
        filename = '%s/images/%s_%s.jpg' % (self.data_path, tomo_id, slice_number)
        img = Image.open(filename)
        img = np.array(img).astype(np.float32) / 255

        return img

    def load_label(self, tomo_id, slice_number):
        """
        Returns
          label (array[float32]): xy (1, 2)
        """
        filename = '%s/labels/%s_%s.txt' % (self.data_path, tomo_id, slice_number)
        a = np.loadtxt(filename, ndmin=2)
        assert a.shape == (1, 2)

        return a

    def dataset(self, idx, augment):
        df = self.df.iloc[idx] if idx is not None else self.df
        slices = create_slices(df, self.datad)
        return Dataset(slices, self, augment=augment)

    def loader(self, idx, cfg_loader, *, shuffle=False, augment=False):
        batch_size = cfg_loader['batch_size']
        num_workers = cfg_loader['num_workers']

        ds = self.dataset(idx, augment)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                            shuffle=shuffle, drop_last=shuffle)
        return loader

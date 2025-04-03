
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms.functional as TTF
from torch.utils.data import DataLoader


def get_tomos(input_path: Path, data_type: str, *, n=None) -> list[Path]:
    """
    Args
      input_path (Path): Kaggle input directory
      data_type (str):   train or test
      n (Optional[int]): reduce tomo_paths to first n
    """
    data_path = input_path / data_type
    tomo_paths = sorted(data_path.glob('*'))

    if n is not None:
        tomo_paths = tomo_paths[:n]

    return tomo_paths


def create_df(input_path: Path):
    """
    Create DataFrame of tomo_ids for KFold split

    Returns:
      df (pd.DataFrame): tomo_ids
      datad (dict): tomo_id -> list of slice numbers
    """
    tomo_paths = get_tomos(input_path, 'train')

    rows = []
    for tomo_path in tomo_paths:
        row = {'tomo_id': tomo_path.name,
               'tomo_dir': str(tomo_path)}
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def preprocess(img: torch.Tensor) -> torch.Tensor:
    """
    Resize and normalize

    Arg:
      img (Tensor[uint8]):   (batch_size, C, H, W)

    Returns:
      img (Tensor[float32]): (batch_size, C, size, size); size = 640
    """
    size = (640, 640)

    img = img.to(dtype=torch.float32)
    img = TTF.resize(img, size)  # (batch_size, C, size, size)

    batch_size, nch, h, w = img.shape
    q = torch.Tensor([0.05, 0.95]).to(img.device)
    x_min, x_max = torch.quantile(img.view(batch_size, nch * h * w), q, dim=1)
    x_min = x_min.view(batch_size, 1, 1, 1)
    x_max = x_max.view(batch_size, 1, 1, 1)

    img = (img - x_min) / (x_max - x_min)
    img = torch.clamp(img, 0, 1)

    return img


class Dataset(torch.utils.data.Dataset):
    """
    dataset = Dataset(tomo_path)

    Args:
      tomo_path (Path): directory including jpg images
    """
    def __init__(self, tomo_path: Path):
        self.filenames = sorted(tomo_path.glob('*'))  # list[Path]
        self.image_shape = None

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> dict:
        filename = self.filenames[i]  # Path
        filebase = filename.stem
        assert filebase[:6] == 'slice_'
        slice_number = int(filebase[6:])  # slice_0000 -> int(0000)

        # Load, resize and normalize image
        img = Image.open(filename)
        W, H = img.size
        img = np.expand_dims(np.array(img), axis=0)  # array[uint8] (1, H, W)

        ret = {'img': img,
               'slice_number': slice_number,
               'shape': np.array((H, W), dtype=int),  # original shape H, W
        }

        return ret

    def loader(self, batch_size: int, num_workers: int):
        loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers)
        return loader

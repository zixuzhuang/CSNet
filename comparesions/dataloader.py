import pickle
from glob import glob
from os.path import join

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from utils.Config import Config
from utils.Result import Result


class Data(object):
    def __init__(self) -> None:
        super().__init__()
        self.image = None
        self.label = None
        return

    def to(self, device):
        self.image = self.image.to(device)
        self.label = self.label.to(device)
        return


class EuclideanDataset(Dataset):
    def __init__(self, data_type, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.f = np.loadtxt(f"{cfg.index_folder}/{data_type}_{cfg.fold}.csv", dtype=str)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        data = np.load(f"{self.cfg.path}/{self.f[idx]}", allow_pickle=True)
        image = torch.tensor(data["image"], dtype=torch.float32)
        label = torch.tensor(self["label"], dtype=torch.long)
        return image, label


def EuclideanDataloader(cfg: Config):
    dataset = {}
    type_list = ["train", "valid", "test"]
    for item in type_list:
        dataset[item] = DataLoader(
            EuclideanDataset(cfg, data_type=item, fold=cfg.fold),
            batch_size=cfg.bs,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate,
        )
    return dataset


def collate(samples):
    _data = Data()
    image, label = map(list, zip(*samples))
    _data.image = torch.stack(image, dim=0)[:, None, :, :]
    _data.label = torch.tensor(label)
    return _data

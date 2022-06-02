import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils.Config import Config


class CSNetData(object):
    def __init__(self):
        super().__init__()
        self.patch = None
        self.pos = None
        self.graph = None
        self.label = None
        return

    def to(self, device):
        self.patch = self.patch.to(device)
        self.pos = self.pos.to(device)
        self.graph = self.graph.to(device)
        self.label = self.label.to(device)
        return


def collate(samples):
    _data = CSNetData()
    graphs, patch, label, pos = map(list, zip(*samples))
    _data.label = torch.tensor(label)
    _data.patch = torch.cat(patch, dim=0)[:, None, :, :]
    _data.pos = torch.cat(pos, dim=0)
    _data.graph = dgl.batch(graphs)
    return _data


def CSNetDataloader(cfg: Config):
    dataset = {}
    type_list = ["train", "valid", "test"]
    for item in type_list:
        dataset[item] = DataLoader(
            CSNetDataset(data_type=item, cfg=cfg),
            batch_size=cfg.bs,
            collate_fn=collate,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
    return dataset


class CSNetDataset(Dataset):
    def __init__(self, data_type: str, cfg: Config):
        super(CSNetDataset, self).__init__()
        self.cfg = cfg
        self.f = np.loadtxt(f"{cfg.index_folder}/{data_type}_{cfg.fold}.csv", dtype=str)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        data = np.load(f"{self.cfg.path}/{self.f[idx]}", allow_pickle=True)
        graph = data["graph"].item()
        patch = torch.tensor(data["patch"], dtype=torch.float32)
        label = torch.tensor(data["label"], dtype=torch.long)
        pos = torch.tensor(data["pos"], dtype=torch.float32)
        return (graph, patch, label, pos)

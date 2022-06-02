import logging
import time

import numpy as np
import torch
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, roc_auc_score)

from utils.Config import Config


def get_one_hot(label, num_cls):
    label = label.reshape(-1)
    label = np.eye(num_cls)[label]
    return label


class Result(object):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.epoch = 1
        self.best_epoch = 0
        self.best_result = 0.0
        return

    def add(self, pred, true):
        self.preds.append(pred)
        self.trues.append(true)
        return

    def init(self):
        self.st = time.time()
        self.preds = []
        self.trues = []
        return

    def stastic(self):
        self.preds = torch.cat(self.preds, dim=0)
        self.trues = torch.cat(self.trues, dim=0)

        probe = torch.softmax(self.preds, dim=1)
        true = self.trues.cpu().detach().numpy()
        pred = probe.cpu().detach().numpy()
        preds = np.argmax(pred, axis=1)
        true_one_hot = get_one_hot(true, self.cfg.num_cls)

        self.acc = accuracy_score(true, preds)
        self.rec = sensitivity_score(true, preds, average="macro")
        self.spe = specificity_score(true, preds, average="macro")
        self.pre = precision_score(true, preds, average="macro", zero_division=0)
        self.f1 = f1_score(true, preds, average="macro")
        self.auc = roc_auc_score(true_one_hot, pred, average="macro")
        self.cm = confusion_matrix(true, preds)
        self.time = np.round(time.time() - self.st, 1)

        self.pars = [self.acc, self.rec, self.spe, self.pre, self.f1, self.auc]
        self.pars = [np.round(_, 3) for _ in self.pars]
        return

    def print(self, datatype: str, epoch: int):
        titles = ["dataset", "ACC", "REC", "SPE", "PRE", "F1S", "AUC"]
        items = [datatype.upper()] + self.pars
        forma_1 = "\n|{:^8}" + "|{:^5}" * (len(titles) - 1) + "|"
        forma_2 = "\n|{:^8}" + "|{:^.3f}" * (len(titles) - 1) + "|"
        logging.info(f"{datatype.upper()} REC: {self.pars[1]}, TIME: {self.time}s")
        logging.info((forma_1 + forma_2).format(*titles, *items))
        logging.debug(f"\n{self.cm}")
        self.epoch = epoch

        if self.rec > self.best_result:
            self.best_epoch = epoch
            self.best_result = self.rec
        return

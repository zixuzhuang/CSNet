import os
import time

import torch
import yaml


class Config(object):
    def __init__(self, args) -> None:
        super().__init__()
        cfg = yaml.load(open(args.c), Loader=yaml.FullLoader)

        # Training Settings
        self.num_workers = cfg["num_workers"]
        self.bs = cfg["bs"]
        self.fold = args.f
        self.test = args.t
        self.device = torch.device("cuda")

        # Data Settings
        self.path = cfg["path"]
        self.index_folder = cfg["index_folder"]
        self.result = cfg["result"]
        self.num_cls = cfg["num_cls"]

        # Optimizer settings
        self.lr = cfg["lr"]
        self.momentum = cfg["momentum"]
        self.weight_decay = cfg["weight_decay"]
        self.lr_freq = cfg["lr_freq"]
        self.lr_decay = cfg["lr_decay"]

        # Model Settings
        self.task = cfg["task"]
        self.net = cfg["net"]
        self.input_size = cfg["input_size"]
        self.num_epoch = cfg["num_epoch"]

        try:
            cfg["pretrain"][0]
            if os.path.exists(cfg["pretrain"][0]):
                self.pretrain = cfg["pretrain"]
            else:
                self.pretrain = None
        except:
            pass

        self.TIME = time.strftime("%Y-%m-%d-%H-%M")  # time of we run the script

        if self.test:
            self.path_log = os.path.join("results", "temp")
            self.path_ckpt = self.path_log
            self.log_dir = os.path.join(self.path_log, "test.log")
            self.best_ckpt = os.path.join(self.path_ckpt, "best.pth")
            self.last_ckpt = os.path.join(self.path_ckpt, "last.pth")
        else:
            self.path_log = os.path.join(self.result, "logs", self.task, self.net)
            self.path_ckpt = os.path.join(self.result, "checkpoints", self.task, self.net)
            self.log_dir = os.path.join(self.path_log, "{}-{}.log".format(self.fold, self.TIME))
            self.best_ckpt = os.path.join(self.path_ckpt, "{}-{}-{}".format(self.fold, self.TIME, "best.pth"))
            self.last_ckpt = os.path.join(self.path_ckpt, "{}-{}-{}".format(self.fold, self.TIME, "last.pth"))

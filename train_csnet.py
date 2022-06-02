import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from CSNet.csnet import CSNet
from CSNet.dataloader import CSNetDataloader
from CSNet.encoder import Encoder
from utils.Config import Config
from utils.Result import Result
from utils.utils_net import get_lr, init_train, save_model


def train(epoch):

    st = time.time()
    running_loss = 0.0
    net.train()

    for data in dataset["train"]:
        data.to(cfg.device)
        optimizer.zero_grad()
        preds, _ = net(data)
        loss = loss_function(preds, data.label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    ft = time.time()
    epoch_loss = running_loss / len(dataset["train"])
    logging.info(f"\n\nEPOCH: {epoch}")
    logging.info(f"TRAIN_LOSS : {epoch_loss:.3f}, TIME: {ft - st:.1f}s")
    return epoch_loss


@torch.no_grad()
def eval(datatype, epoch):
    r = result[datatype]
    r.init()
    net.eval()
    for data in dataset[datatype]:
        data.to(cfg.device)
        preds, _ = net(data)
        r.add(preds, data.label)
    r.stastic()
    r.print(datatype, epoch)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("-c", type=str, default="CSNet/csnet_subject.yaml")
    parser.add_argument("-t", type=bool, default=False)
    args = parser.parse_args()

    cfg = Config(args)
    init_train(cfg)
    encoders = Encoder().to(cfg.device)
    net = CSNet(encoders).to(cfg.device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    dataset = CSNetDataloader(cfg)
    result = {"valid": Result(cfg), "test": Result(cfg)}

    for _epoch in range(cfg.num_epoch):

        epoch = _epoch + 1
        # Learning rate decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(epoch, cfg)

        # Train
        train(epoch)

        # Evaluation
        eval("valid", epoch)
        eval("test", epoch)

        # Save
        save_model(epoch, result, net, cfg)

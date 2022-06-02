import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from comparesions.dataloader import EuclideanDataloader
from comparesions.MedT3D.medt import get_args, medT
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils.Config import Config
from utils.Result import Result
from utils.utils_net import get_lr, init_train, save_model


def train(epoch):

    st = time.time()
    running_loss = 0.0
    net.train()

    for images, labels in tqdm(dataset["train"], desc="iter", position=1, leave=False):
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)
        optimizer.zero_grad()
        with autocast():
            cls = net(images)
            loss = loss_function(cls, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss
    ft = time.time()
    epoch_loss = running_loss / len(dataset["train"])
    logging.info("EPOCH: {}".format(epoch))
    logging.info("TRAIN_LOSS : {:.3f}, TIME: {:.1f}s".format(epoch_loss, ft - st))
    return epoch_loss


@torch.no_grad()
def eval(datatype):

    result[datatype].init()
    net.eval()
    for images, labels in dataset[datatype]:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)
        with autocast():
            cls = net(images)
        result[datatype].add(cls, labels)
    result[datatype].stastic()
    result[datatype].print()
    return


if __name__ == "__main__":

    args = get_args(argparse.ArgumentParser())
    scaler = GradScaler()
    cfg = Config(args)
    init_train(cfg)
    loss_function = nn.CrossEntropyLoss()
    net = medT(cfg.device).to(cfg.device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    dataset = EuclideanDataloader(cfg)
    result = {}
    result["valid"], result["test"] = Result(cfg), Result(cfg)

    for epoch in range(1, cfg.num_epoch):

        # Learning rate decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(epoch, cfg)

        # Train
        loss = train(epoch)

        # Calculate val data and test data and save failed file
        eval("valid")
        eval("test")

        # Save  model
        save_model(epoch, result, net, cfg)

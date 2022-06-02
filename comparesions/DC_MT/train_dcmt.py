import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from comparesions.dataloader import EuclideanDataloader
from comparesions.DC_MT.dcmt import FABlock, MeanTeacherNet, get_args
from tqdm import tqdm
from utils.Config import Config
from utils.Result import Result
from utils.utils_net import get_lr, init_train, save_model


def train(epoch):

    st = time.time()
    running_loss = 0.0
    net.train()

    for data in tqdm(dataset["train"], desc="iter", position=1, leave=False):
        data.to(cfg.device)
        optimizer.zero_grad()
        s_y, s_fm = net.student(data.image)
        with torch.no_grad():
            t_y, t_fm = net.teacher(data.image)
        s_cam = cam(s_fm)[1]
        t_cam = cam(t_fm)[1]
        loss = CEloss(s_y, data.label) + 0.5 * MSEloss(s_cam, t_cam) + 0.001 * MSEloss(s_y, t_y)
        loss.backward()
        optimizer.step()
        net.ema_update()
        running_loss += loss
    ft = time.time()
    epoch_loss = running_loss / len(dataset["train"])
    logging.info("EPOCH: {}".format(epoch))
    logging.info("TRAIN_LOSS : {:.3f}, TIME: {:.1f}s".format(epoch_loss, ft - st))
    return epoch_loss


@torch.no_grad()
def eval(datatype):

    st = time.time()
    result[datatype].init()
    net.eval()
    for data in dataset[datatype]:
        data.to(cfg.device)
        t_y, t_fm = net.teacher(data.image)
        result[datatype].add(t_y, data.label)
    result[datatype].stastic()
    result[datatype].print()
    ft = time.time()
    result[datatype].time = ft - st
    return


if __name__ == "__main__":

    args = get_args(argparse.ArgumentParser())
    cfg = Config(args)
    init_train(cfg)
    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()
    net = MeanTeacherNet().to(cfg.device)
    cam = FABlock(512).to(cfg.device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    dataset = EuclideanDataloader(cfg)
    result = {"valid": Result(cfg), "test": Result(cfg)}

    for _epoch in range(1, cfg.num_epoch):

        epoch = _epoch + 1

        # Learning rate decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(epoch, cfg)

        # Train
        train(epoch)

        # Calculate val data and test data and save failed file
        eval("valid")
        eval("test")

        # Save  model
        save_model(epoch, result, net, cfg)

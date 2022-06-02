import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from comparesions.dataloader import EuclideanDataloader
from comparesions.UNet_Encoder.segcls import UNet, get_args
from utils.Config import Config
from utils.utils_net import get_lr, init_train


def train(epoch):

    st = time.time()
    running_loss = 0.0
    net.train()

    for batch_index, data in enumerate(dataloader["train"]):
        data.to(cfg.device)
        optimizer.zero_grad()
        bones = net(data.image)
        loss = loss_function(bones, data.label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    ft = time.time()
    epoch_loss = running_loss / len(dataloader["train"])
    logging.info("EPOCH: {}".format(epoch))
    logging.info("TRAIN_LOSS : {:.3f}, TIME: {:.1f}s".format(epoch_loss, ft - st))
    return epoch_loss


@torch.no_grad()
def eval_training(epoch, datatype):

    st = time.time()
    net.eval()
    loss = 0.0
    for data in dataloader[datatype]:
        data.to(cfg.device)
        bones = net(data.image)
        loss += loss_function(bones, data.label).item()
    ft = time.time()

    return loss / len(dataloader[datatype]), ft - st


if __name__ == "__main__":

    args = get_args(argparse.ArgumentParser())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    cfg = Config(args)
    loss_function = nn.CrossEntropyLoss()

    init_train(cfg)
    net = UNet().to(cfg.device)
    optimizer = optim.Adam(net.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    dataloader = EuclideanDataloader(cfg)
    best_loss = 100000.0
    best_epoch = 0
    for epoch in range(1, cfg.num_epoch + 1):

        # Learning rate decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(epoch, cfg)

        # Train
        train(epoch)

        # Calculate val data and test data and save failed file
        v_loss, v_time = eval_training(epoch, "val")
        t_loss, t_time = eval_training(epoch, "test")
        logging.info("TRAI LOSS: {:.3f}, TIME: {:.1f}s".format(v_loss, v_time))
        logging.info("TEST LOSS: {:.3f}, TIME: {:.1f}s".format(t_loss, t_time))

        # Save  model
        if not os.path.exists(cfg.path_ckpt):
            os.makedirs(cfg.path_ckpt)

        # Save best model
        if best_loss > t_loss:
            best_loss = t_loss
            best_epoch = epoch
            torch.save(net, cfg.best_ckpt)
        logging.info("BEST LOSS: {:.3f}, EPOCH: {:3}\n\n".format(best_loss, best_epoch))

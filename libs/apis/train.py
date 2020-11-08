# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.


import time
import logging
import os.path as osp

import torch
import torch.nn as nn

from utils import build_lr_scheduler
from .function import train, valid


logger = logging.getLogger(__name__)



def train_model(configs, model, train_loader, valid_loader):
    criterion = nn.L1Loss().cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=configs.optimizer.lr,
                                 betas=configs.optimizer.betas,
                                 eps=configs.optimizer.eps)

    scheduler = build_lr_scheduler(configs.lr_config, optimizer)

    total_epochs = configs.total_epoches

    best = 0
    best_model_path = osp.join(configs.work_dir, "best.pth")

    for i in range(total_epochs):
        train(i, configs, model, train_loader, optimizer, criterion)
        perf_indicator = valid(configs, model, valid_loader)
        if best < perf_indicator['psnr']:
            logger.info("=>Save best model to {}".format(best_model_path))
            torch.save(model.state_dict(), best_model_path)
            best = perf_indicator['psnr']
        scheduler.step()

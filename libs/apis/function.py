# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import logging
logger = logging.getLogger(__name__)

import os.path as osp

import cv2
import time
import torch
import numpy as np

from .utils import AverageMeter, psnr, SSIM



def train(epoch, configs, model, train_loader, optimizer, criterion):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time()-end)
        hr, lr = batch
        hr = hr.cuda()
        lr = lr.cuda()

        output = model(lr)
        optimizer.zero_grad()
        loss = criterion(output, hr)
        loss.backward()

        losses.update(loss.data, lr.size(0))
        optimizer.step()

        batch_time.update(time.time() - end)

        if i % configs.log_config.interval == 0:
            msg = 'Epoch: [{:3}][{:2}/{:2}] ' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s) ' \
                  'Speed {speed:.1f} samples/s ' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s) ' \
                  'Loss {loss.val:.3f} ({loss.avg:.3f}) '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        speed=lr.size(0)/batch_time.val, data_time=data_time, loss=losses)
            logger.info(msg)
        end = time.time()

def valid(configs, model, valid_loader):
    
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    perf_indicator = dict()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            hr, lr = batch
            hr, lr = hr.cuda(), lr.cuda()

            output = model(lr)

            psnr_meter.update(psnr(output, hr))

        msg = 'PSNR: {psnr.val:.3f}({psnr.avg:.3f})'.format(psnr=psnr_meter)
        logger.info(msg)
    perf_indicator['psnr'] = psnr_meter.avg
    return perf_indicator
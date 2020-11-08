# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import os.path as osp
import time

import argparse
import torch
import mmcv
from mmcv import Config
from torch.utils.data import DataLoader

import _init_path

import models
import datasets
from apis.train import train_model
from utils.logger import create_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file')
    parser.add_argument('--work_dir', help='save model and logfile')

    return parser.parse_args()


def main():
    args = parse_args()
    configs = Config.fromfile(args.config)
    if args.work_dir is not None:
        configs.work_dir = args.work_dir

    mmcv.mkdir_or_exist(configs.work_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(configs.work_dir, '{}.log'.format(timestamp))
    logger = create_logger(log_file=log_file)

    logger.info(configs.text)

    dataset_config = configs.data
    train_dataset = eval('datasets.{}.build_dataset'.format(dataset_config.train.type)) \
        (dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=dataset_config.batch,
                              shuffle=True, num_workers=dataset_config.num_workers)

    valid_dataset = eval('datasets.{}.build_dataset'.format(dataset_config.valid.type)) \
        (dataset_config, is_training=False)
    valid_loader = DataLoader(valid_dataset, batch_size=dataset_config.batch,
                              shuffle=False, num_workers=dataset_config.num_workers)

    model = eval('models.{}.build_model'.format(configs.model.name))(configs)
    model = torch.nn.DataParallel(model, device_ids=configs.gpu_group).cuda()

    train_model(configs, model, train_loader, valid_loader)


if __name__ == '__main__':
    main()

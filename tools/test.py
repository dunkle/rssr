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
from apis.valid import valid_model
from utils.logger import create_logger



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('config', help='model config file ')
    parser.add_argument('model', help='model path')
    parser.add_argument('--eval', help='must be ssim or psnr', default='psnr')
    parser.add_argument('--work_dir', help='save test log file')


    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    configs = Config.fromfile(args.config)
    if args.work_dir is not None:
        configs.work_dir = args.work_dir

    assert osp.exists(args.model), "Model {} not exists".format(args.model)

    logger = create_logger()
    logger.info(configs.text)

    dataset_config = configs.data 
    test_dataset = eval('datasets.{}.build_dataset'.format(dataset_config.test.type))(dataset_config, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=dataset_config.batch,
                            shuffle=True, num_workers=dataset_config.num_workers)


    model = eval('models.{}.build_model'.format(configs.model.name))(configs)
    model = torch.nn.DataParallel(model, device_ids=configs.gpu_group).cuda()
    model.load_state_dict(torch.load(args.model))

    # Test model
    valid_model(configs, model, test_loader)
    

if __name__ == "__main__":
    main()


# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import argparse
from mmcv.utils import Config
from torch.utils.data import DataLoader
import torch.nn.functional as F

import _init_path
import datasets
from utils import AverageMeter, create_logger, setup_seed
from apis import psnr


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file')
    parser.add_argument('--seed', type=int, help="random seed")

    return parser.parse_args()


def main():
    args = parser_args()
    if args.seed != None:
        setup_seed(args.seed)

    config = Config.fromfile(args.config)
    logger = create_logger()

    dataset_config = config.data
    dataset = eval('datasets.{}.build_dataset'.format(dataset_config.test.type)) \
        (dataset_config, is_training=False)
    loader = DataLoader(dataset, batch_size=1,
                        shuffle=True, num_workers=dataset_config.num_workers)
    psnr_meter = AverageMeter()
    for i, batch in enumerate(loader):
        hr, lr = batch
        bicubic_hr = F.interpolate(lr, scale_factor=dataset_config.scale, mode='bicubic', align_corners=True)

        psnr_meter.update(psnr(hr, bicubic_hr))

    logger.info('PSNR for {} is {psnr.val:.3f}'.format(dataset_config.test.type, psnr=psnr_meter))


if __name__ == '__main__':
    main()


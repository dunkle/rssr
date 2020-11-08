# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import logging

logger = logging.getLogger(__name__)

import os
import os.path as osp
import glob

import cv2
import numpy as np
from torch.utils.data import Dataset

from .utils import get_patch


class CustomDataset(Dataset):
    """

    Args:
        factor: (int): downsample factor of hr to lr
        data_root: (str): Data root for lr_dir and hr_dir
    """

    def __init__(self, data_root,
                 lr_dir, hr_dir,
                 scale=4, patch_size=64,
                 crop_image=True,
                 is_training=True,
                 img_ext='jpg'):
        self.scale = scale
        self.crop_image = crop_image
        self.patch_size = patch_size
        self.data_root = data_root
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.is_training = is_training
        self.img_ext = img_ext
        self.images = []
        self._get_images()
        self.transformer = None

        logger.info("Root Path is {}".format(osp.join(self.data_root, self.hr_dir)))
        logger.info("Load {} images".format(len(self.images)))

    def _get_images(self):
        self.images = glob.glob(osp.join(self.data_root, self.hr_dir, "*.{}".format(self.img_ext)))


    def _get_hr_lr_path(self, index):
        hr_path = self.images[index]
        hr_name = osp.basename(hr_path)
        lr_name = "{}_x{}.{}".format(hr_name.split('.')[0], self.scale, self.img_ext)
        lr_path = osp.join(self.data_root, self.lr_dir, lr_name)

        return hr_path, lr_path

    def __getitem__(self, index):
        hr_path, lr_path = self._get_hr_lr_path(index)

        assert osp.exists(hr_path), '{} doesn\'t exits'.format(hr_path)
        assert osp.exists(lr_path), '{} doesn\'t exits'.format(lr_path)

        hr_image = cv2.imread(hr_path)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.imread(lr_path)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        if self.transformer != None:
            hr_image, lr_image = self.transformer(hr_image, lr_image)

        if not self.is_training:
            hr_image = hr_image.transpose((2, 0, 1)).astype(np.float32)
            lr_image = lr_image.transpose((2, 0, 1)).astype(np.float32)
            return hr_image, lr_image

        if self.crop_image:
            hr_patch, lr_patch = get_patch(hr_image, lr_image, self.scale, self.patch_size)
        else:
            hr_patch, lr_patch = hr_image, lr_image

        hr_patch = hr_patch.transpose((2, 0, 1)).astype(np.float32)
        lr_patch = lr_patch.transpose((2, 0, 1)).astype(np.float32)

        return hr_patch, lr_patch

    def __len__(self):
        return len(self.images)


def build_dataset(config, is_training=True):
    dataset_config = config.train if is_training else config.valid
    dataset = MunichDataset(dataset_config.img_prefix,
                            dataset_config.lr_dir,
                            dataset_config.hr_dir,
                            scale=config.scale,
                            patch_size=dataset_config.patch_size if is_training else 0,
                            is_training=is_training)

    return dataset

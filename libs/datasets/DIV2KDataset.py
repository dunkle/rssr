# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import logging

logger = logging.getLogger(__name__)

import os.path as osp
from .CustomDataset import CustomDataset


class DIV2KDataset(CustomDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_hr_lr_path(self, index):
        hr_path = self.images[index]
        hr_name = osp.basename(hr_path)
        lr_name = "{}x{}.{}".format(hr_name.split('.')[0], self.scale, self.img_ext)
        lr_path = osp.join(self.data_root, self.lr_dir, 'X{}'.format(self.scale), lr_name)

        return hr_path, lr_path


def build_dataset(config, is_training=True):
    dataset_config = config.train if is_training else config.valid
    dataset = DIV2KDataset(dataset_config.img_prefix,
                           dataset_config.lr_dir,
                           dataset_config.hr_dir,
                           scale=config.scale,
                           patch_size=dataset_config.patch_size if is_training else 0,
                           is_training=is_training,
                           img_ext='png')

    return dataset

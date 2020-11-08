# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import logging
logger = logging.getLogger(__name__)

from torch.utils.data import Dataset
from .CustomDataset import CustomDataset


class MunichDataset(CustomDataset):
    """

    Args:
        factor: (int): downsample factor of hr to lr
        data_root: (str): Data root for lr_dir and hr_dir
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def build_dataset(config, is_training=True):
    dataset_config = config.train if is_training else config.valid
    dataset = MunichDataset(dataset_config.img_prefix,
                            dataset_config.lr_dir,
                            dataset_config.hr_dir,
                            scale=config.scale,
                            patch_size=dataset_config.patch_size if is_training else 0,
                            is_training=is_training)

    return dataset

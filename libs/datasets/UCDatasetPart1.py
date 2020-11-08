# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import cv2
import os.path as osp
import glob
from .CustomDataset import CustomDataset
from torch.utils.data import Dataset
from .utils import ResizeTransformer

import logging
logger = logging.getLogger(__name__)






class UCDatasetPart1(CustomDataset):
    """

    Args:
        factor: (int): downsample factor of hr to lr
        data_root: (str): Data root for lr_dir and hr_dir
    """

    PART1_CLASSES = ("airplane", "buildings", "denseresidential",
                     "freeway", "harbor", "intersection", "overpass",
                     "parkinglot", "storagetanks", "tenniscourt")


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = ResizeTransformer(hr_size=(256, 256), scale=self.scale)

    def _get_images(self):
        for class_name in self.PART1_CLASSES:
            hr_root = osp.join(self.data_root, self.hr_dir, 
                            class_name)
            lr_root = osp.join(self.data_root, self.lr_dir,  "x{}".format(self.scale),
                            class_name)
            
            for hr_image in glob.glob(osp.join(hr_root,"*.{}".format(self.img_ext))):
                imgid = osp.basename(hr_image).split('.')[0]
                self.images.append((hr_image, 
                                    osp.join(lr_root, 
                                            "{}_x{}.{}".format(imgid, self.scale, self.img_ext))))

    def _get_hr_lr_path(self, index):
        return self.images[index]



def build_dataset(config, is_training=True):
    """
    这里输入的config是config中的data，需要根据是否为test选择config.train还是config.valid还是config.test
    """
    dataset_config = config.train if is_training else config.valid
    dataset = UCDatasetPart1(dataset_config.img_prefix,
                             dataset_config.lr_dir,
                             dataset_config.hr_dir,
                             scale=config.scale,
                             crop_image=False,
                             patch_size=dataset_config.patch_size if is_training else 0,
                             is_training=is_training, img_ext='tif')

    return dataset

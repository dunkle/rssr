# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import random
import numpy as np

import cv2
from typing import Tuple


def get_patch(hr: np.ndarray,
              lr: np.ndarray,
              factor, patch_size,
              start_x=-1, start_y=-1):
    """

    :param hr: (np.ndarry) High resolution image
    :param lr: (np.ndarry) Low resolution image
    :param factor: down sample factor
    :param patch_size: patch size of cropped image used to train
    :param start_x: start x coordinate of patch
    :param start_y: start y coordinate of patch
    :return:
    """
    lr_shape = lr.shape
    if start_x == -1:
        start_x = random.randrange(0, lr_shape[1] - patch_size + 1)
    if start_y == -1:
        start_y = random.randrange(0, lr_shape[2] - patch_size + 1)

    hr_start_x = start_x * factor
    hr_start_y = start_y * factor
    hr_patch_size = patch_size * factor
    lr_patch = lr[:, start_x:start_x + patch_size, start_y:start_y + patch_size]
    hr_patch = hr[:, hr_start_x:hr_start_x + hr_patch_size,
               hr_start_y:hr_start_y + hr_patch_size]

    return hr_patch, lr_patch

class ResizeTransformer:
    
    def __init__(self, hr_size=(256, 256), scale=4):
        self.hr_size = hr_size
        self.lr_size = (hr_size[0]//scale, hr_size[1]//scale)


    def __call__(self, hr_image, lr_image):
        hr_image = cv2.resize(hr_image, self.hr_size, interpolation=cv2.INTER_CUBIC) 
        lr_image = cv2.resize(lr_image, self.lr_size, interpolation=cv2.INTER_CUBIC) 

        return hr_image, lr_image

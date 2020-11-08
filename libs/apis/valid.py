# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import logging
logger = logging.getLogger(__name__)

import os.path as osp

import torch
import numpy as np
import cv2

from .function import valid



def valid_model(configs, model, test_loader):
    perf_indicator = valid(configs, model, test_loader) 

    
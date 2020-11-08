# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

from torch.optim.lr_scheduler import MultiStepLR, StepLR

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def build_lr_scheduler(lr_config, optimizer):
    if lr_config.policy == 'linear':
        scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    elif lr_config.policy == 'step':
        scheduler = StepLR(optimizer, step_size=200)
    else:
        raise NotImplementedError('Lr scheduler {} not found'.format(lr_config.policy))

    return scheduler

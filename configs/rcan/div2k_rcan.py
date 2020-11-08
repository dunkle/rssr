# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.


gpu_group = [0]

data_type = 'DIV2KDataset'
data_root = '/home/data/DIV2K/'
data = dict(
    batch=16,
    num_workers=8,
    scale=2,
    train=dict(
        type=data_type,
        patch_size=64,
        img_prefix=data_root,
        hr_dir='DIV2K_train_HR',
        lr_dir='DIV2K_train_LR'
    ),
    valid=dict(
        type=data_type,
        img_prefix=data_root,
        hr_dir='DIV2K_valid_HR',
        lr_dir='DIV2K_valid_LR'
    ),
    test=dict(
        type=data_type,
        img_prefix=data_root,
        hr_dir='DIV2K_valid_HR',
        lr_dir='DIV2K_valid_LR'
    )
)

model = dict(
    name='rcan',
    pretrained=None,
    num_resgroups=10,
    num_resblocks=20,
    channels=64,
    reduction=16
)

optimizer = dict(type='Adam', lr=1e-4, betas=[0.9, 0.999], eps=1e-8)
lr_config = dict(
    policy='step',
    step_size=200
)

total_epoches = 1000
work_dir = './work_dir/rcan'
log_config = dict(interval=20)

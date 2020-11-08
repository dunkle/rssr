# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.


gpu_group = [0]

data_type = 'MunichDataset'
data_root = '/home/dsp/Data/MunichDatasetVehicleDetection-2015-old/'
data = dict(
    batch=8,
    num_workers=8,
    scale=2,
    train=dict(
        type=data_type,
        patch_size=64,
        img_prefix=data_root + 'SRTrain',
        lr_dir='train_LR',
        hr_dir='train_HR'
    ),
    valid=dict(
        type=data_type,
        img_prefix=data_root + 'SRValid',
        lr_dir='valid_LR',
        hr_dir='valid_HR'
    ),
    test=dict(
        type=data_type,
        img_prefix=data_root + 'SRValid',
        lr_dir='valid_LR',
        hr_dir='valid_HR'
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
    policy='linear',
    step=[30, 60]
)

total_epoches = 100
work_dir = './work_dir/rcan'
log_config = dict(interval=20)

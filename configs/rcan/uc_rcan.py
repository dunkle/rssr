# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.


gpu_group = [0]

part1_type = 'UCDatasetPart1'
part2_type = 'UCDatasetPart2'
data_root = '/home/data/UCMerced_LandUse/Part1/'
data = dict(
    batch=8,
    num_workers=8,
    scale=4,
    train=dict(
        type=part1_type,
        patch_size=64,
        img_prefix=data_root + 'train_data',
        lr_dir='LR',
        hr_dir='HR'
    ),
    valid=dict(
        type=part1_type,
        img_prefix=data_root + 'valid_data',
        lr_dir='LR',
        hr_dir='HR'
    ),
    test=dict(
        type=part2_type,
        img_prefix='/home/data/UCMerced_LandUse/Part2/',
        lr_dir='LR',
        hr_dir='HR'
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

"""
Default Transform
By Zhaofan Qiu, Copyright 2022.
"""
import re

import torch
from torchvision import transforms
from . import transform_func
from .clip_rand_augment import ClipRandAugment


def get_rand_aug_train_transform(cfg):
    auto_augment = cfg.TRANSFORM.AUTO_AUG
    config = auto_augment.split('-')
    n = 1
    m = 27
    assert config[0] == "clip_rand"
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "m":
            m = int(val)
        if key == "n":
            n = int(val)
        else:
            assert NotImplementedError

    aug_transform = transforms.Compose([
        transform_func.ClipRandomResizedCrop(cfg.TRANSFORM.CROP_SIZE, scale=(0.2, 1.),
                                             ratio=(0.75, 1.3333333333333333)),
        ClipRandAugment(n=n, m=m),
        transform_func.ClipRandomHorizontalFlip(p=0.0 if cfg.TRANSFORM.NO_HORIZONTAL_FLIP else 0.5),
        transform_func.ToClipTensor(),
        transform_func.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(
            lambda clip: torch.stack(clip, dim=1)) if cfg.TRANSFORM.TIME_DIM == "T" else transforms.Lambda(
            lambda clip: torch.cat(clip, dim=0))
        ])

    return aug_transform

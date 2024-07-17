"""
hacked together by Zhaofan Qiu, Copyright 2022.
"""

from .default_transform import get_default_train_transform, get_default_test_transform
from .mvit_transform import get_mvit_train_transform
from .sifar_transform import get_sifar_train_transform, get_sifar_randaug_train_transform, get_sifar_dynaaug_train_transform, get_sifar_dynaaug_train_transform_exp, get_sifar_dynaaug_train_transform_exp2
from .points_transform import get_points_train_transform, get_points_test_transform
from .rand_aug_transform import get_rand_aug_train_transform

transform_dict = {
    'DefaultTransform': get_default_train_transform,
    'MViTTransform': get_mvit_train_transform,
    'SIFARTransform': get_sifar_train_transform,
    'SIFARRandAugTransform': get_sifar_randaug_train_transform,
    'SIFARDynaAugTransform': get_sifar_dynaaug_train_transform,
    'SIFARDynaAugExpTransform': get_sifar_dynaaug_train_transform_exp,
    'SIFARDynaAugExp2Transform': get_sifar_dynaaug_train_transform_exp2,
    'PointsTransform': get_points_train_transform,
    'RandAugTransform': get_rand_aug_train_transform
}


def build_train_transform(cfg):
    transform = transform_dict.get(cfg.TRANSFORM.NAME)(cfg)
    return transform


def build_test_transform(cfg, crop_idx=0):
    if cfg.TRANSFORM.NAME == 'PointsTransform':
        return get_points_test_transform(cfg, crop_idx)
    else:
        return get_default_test_transform(cfg, crop_idx)

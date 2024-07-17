import multiprocessing
from typing import Union, List, Tuple
from PIL import Image

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from .rand_augment import rand_augment_transform
from .dyna_augment import dyna_augment_transform
from .dyna_augment_exp import dyna_augment_transform_exp
from .dyna_augment_exp2 import dyna_augment_transform_exp2
from .transform_func import (GroupRandomHorizontalFlip, GroupOverSample,
                             GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                             GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def get_sifar_train_transform(cfg):
    image_size = cfg.TRANSFORM.CROP_SIZE
    scale_range = [256, 320]
    version = 'v1'
    threed_data = True if cfg.TRANSFORM.TIME_DIM == 'T' else False

    augments = []

    if version == 'v1':
        augments += [
            GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
        ]
    elif version == 'v2':
        augments += [
            GroupRandomScale(scale_range),
            GroupRandomCrop(image_size),
        ]
    if not cfg.TRANSFORM.NO_HORIZONTAL_FLIP:
        augments += [GroupRandomHorizontalFlip(is_flow=cfg.TRANSFORM.USE_FLOW)]

    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor

def get_sifar_randaug_train_transform(cfg):
    auto_augment = cfg.TRANSFORM.AUTO_AUG
    image_size = cfg.TRANSFORM.CROP_SIZE
    aa_params = {"translate_const": int(image_size * 0.45)}
    aa_params["interpolation"] = Image.BICUBIC
    scale_range = [256, 320]
    version = 'v1'
    threed_data = True if cfg.TRANSFORM.TIME_DIM == 'T' else False

    augments = []

    if version == 'v1':
        augments += [
            GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
        ]
    elif version == 'v2':
        augments += [
            GroupRandomScale(scale_range),
            GroupRandomCrop(image_size),
        ]
    augments += [rand_augment_transform(auto_augment, aa_params)]
    if not cfg.TRANSFORM.NO_HORIZONTAL_FLIP:
        augments += [GroupRandomHorizontalFlip(is_flow=cfg.TRANSFORM.USE_FLOW)]

    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor


def get_sifar_dynaaug_train_transform(cfg):
    auto_augment = cfg.TRANSFORM.AUTO_AUG
    image_size = cfg.TRANSFORM.CROP_SIZE
    aa_params = {"translate_const": int(image_size * 0.45)}
    aa_params["interpolation"] = Image.BICUBIC
    scale_range = [256, 320]
    version = 'v1'
    threed_data = True if cfg.TRANSFORM.TIME_DIM == 'T' else False

    augments = []

    if version == 'v1':
        augments += [
            GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
        ]
    elif version == 'v2':
        augments += [
            GroupRandomScale(scale_range),
            GroupRandomCrop(image_size),
        ]
    augments += [dyna_augment_transform(auto_augment, aa_params)]
    if not cfg.TRANSFORM.NO_HORIZONTAL_FLIP:
        augments += [GroupRandomHorizontalFlip(is_flow=cfg.TRANSFORM.USE_FLOW)]

    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor


def get_sifar_dynaaug_train_transform_exp(cfg):
    auto_augment = cfg.TRANSFORM.AUTO_AUG
    image_size = cfg.TRANSFORM.CROP_SIZE
    aa_params = {"translate_const": int(image_size * 0.45)}
    aa_params["interpolation"] = Image.BICUBIC
    scale_range = [256, 320]
    version = 'v1'
    threed_data = True if cfg.TRANSFORM.TIME_DIM == 'T' else False

    augments = []

    if version == 'v1':
        augments += [
            GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
        ]
    elif version == 'v2':
        augments += [
            GroupRandomScale(scale_range),
            GroupRandomCrop(image_size),
        ]
    augments += [dyna_augment_transform_exp(auto_augment, aa_params)]
    if not cfg.TRANSFORM.NO_HORIZONTAL_FLIP:
        augments += [GroupRandomHorizontalFlip(is_flow=cfg.TRANSFORM.USE_FLOW)]

    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor


def get_sifar_dynaaug_train_transform_exp2(cfg):
    auto_augment = cfg.TRANSFORM.AUTO_AUG
    image_size = cfg.TRANSFORM.CROP_SIZE
    aa_params = {"translate_const": int(image_size * 0.45)}
    aa_params["interpolation"] = Image.BICUBIC
    scale_range = [256, 320]
    version = 'v1'
    threed_data = True if cfg.TRANSFORM.TIME_DIM == 'T' else False

    augments = []

    if version == 'v1':
        augments += [
            GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
        ]
    elif version == 'v2':
        augments += [
            GroupRandomScale(scale_range),
            GroupRandomCrop(image_size),
        ]
    augments += [dyna_augment_transform_exp2(auto_augment, aa_params)]
    if not cfg.TRANSFORM.NO_HORIZONTAL_FLIP:
        augments += [GroupRandomHorizontalFlip(is_flow=cfg.TRANSFORM.USE_FLOW)]

    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor

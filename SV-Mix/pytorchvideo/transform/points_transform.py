"""
Points Transform
By Zhaofan Qiu, Copyright 2022.
"""
import torch
from torchvision import transforms
from . import transform_func
import numpy as np
#import cv2
import PIL
from PIL import Image, ImageFilter, ImageEnhance


def get_points_train_transform(cfg):
    train_transform = transforms.Compose([
        transform_func.GroupMultiScaleCrop(cfg.TRANSFORM.CROP_SIZE, [1, .875, .75, .66]),
        transform_func.ToClipTensor(),
        transform_func.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if cfg.TRANSFORM.TIME_DIM == "T" else transforms.Lambda(
            lambda clip: torch.cat(clip, dim=0))
    ])
    return train_transform


def get_points_test_transform(cfg, crop_idx):
    if crop_idx == 0:
        crop = transform_func.ClipCenterCrop
    elif crop_idx == 1:
        crop = transform_func.ClipFirstCrop
    elif crop_idx == 2:
        crop = transform_func.ClipThirdCrop
    else:
        raise NotImplementedError

    test_transform = transforms.Compose([
        transform_func.ClipResize(size=cfg.TRANSFORM.TEST_CROP_SIZE),
        crop(size=cfg.TRANSFORM.TEST_CROP_SIZE),
        transform_func.ToClipTensor(),
        transform_func.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if cfg.TRANSFORM.TIME_DIM == "T" else transforms.Lambda(
            lambda clip: torch.cat(clip, dim=0))
    ])

    return test_transform

"""
Default Transform
By Zhaofan Qiu, Copyright 2022.
"""
import torch
from torchvision import transforms
from . import transform_func


def get_default_train_transform(cfg):
    if not cfg.TRANSFORM.USE_FLOW:
        train_transform = transforms.Compose([
            transform_func.ClipRandomResizedCrop(cfg.TRANSFORM.CROP_SIZE, scale=(0.2, 1.),
                                                 ratio=(0.75, 1.3333333333333333)),
            transforms.RandomApply([
                transform_func.ClipColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transform_func.ClipRandomGrayscale(p=0.2),
            transforms.RandomApply([transform_func.ClipGaussianBlur([.1, 2.])], p=0.5),
            transform_func.ClipRandomHorizontalFlip(p=0.0 if cfg.TRANSFORM.NO_HORIZONTAL_FLIP else 0.5),
            transform_func.ToClipTensor(),
            transform_func.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if cfg.TRANSFORM.TIME_DIM == "T" else transforms.Lambda(
                lambda clip: torch.cat(clip, dim=0))
        ])
    else:
        train_transform = transforms.Compose([
            transform_func.ClipRandomResizedCrop(cfg.TRANSFORM.CROP_SIZE, scale=(0.5, 1.),
                                                 ratio=(0.75, 1.3333333333333333)),
            transform_func.FlowClipRandomHorizontalFlip(p=0.0 if cfg.TRANSFORM.NO_HORIZONTAL_FLIP else 0.5),
            transform_func.FlowToClipTensor(),
            transform_func.ClipNormalize(mean=[0.5, 0.5], std=[0.229, 0.229]),
            transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if cfg.TRANSFORM.TIME_DIM == "T" else transforms.Lambda(
                lambda clip: torch.cat(clip, dim=0))
        ])
    return train_transform


def get_default_test_transform(cfg, crop_idx):
    if crop_idx == 0:
        crop = transform_func.ClipCenterCrop
    elif crop_idx == 1:
        crop = transform_func.ClipFirstCrop
    elif crop_idx == 2:
        crop = transform_func.ClipThirdCrop
    else:
        raise NotImplementedError

    if not cfg.TRANSFORM.USE_FLOW:
        test_transform = transforms.Compose([
            transform_func.ClipResize(size=cfg.TRANSFORM.TEST_CROP_SIZE),
            crop(size=cfg.TRANSFORM.TEST_CROP_SIZE),
            transform_func.ToClipTensor(),
            transform_func.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if cfg.TRANSFORM.TIME_DIM == "T" else transforms.Lambda(
                lambda clip: torch.cat(clip, dim=0))
        ])
    else:
        test_transform = transforms.Compose([
            transform_func.ClipResize(size=cfg.TRANSFORM.TEST_CROP_SIZE),
            crop(size=cfg.TRANSFORM.TEST_CROP_SIZE),
            transform_func.FlowToClipTensor(),
            transform_func.ClipNormalize(mean=[0.5, 0.5], std=[0.229, 0.229]),
            transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if cfg.TRANSFORM.TIME_DIM == "T" else transforms.Lambda(
                lambda clip: torch.cat(clip, dim=0))
        ])
    return test_transform
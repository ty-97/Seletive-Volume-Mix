"""
Select model, transfer pre-train weights (2D), remove fc layer
By Zhaofan Qiu
zhaofanqiu@gmail.com
"""
import sys
import torch
import logging
from pytorchvideo.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for model
"""

model_dict = {}
transfer_dict = {}


def build_model(cfg):
    model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model

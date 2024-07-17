"""
hacked together by Zhaofan Qiu, Copyright 2022.
"""

import torch
from pytorchvideo.config import configurable
from .build import LR_SCHEDULER_REGISTRY

@LR_SCHEDULER_REGISTRY.register()
class MultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    @configurable
    def __init__(
        self, 
        *,
        optimizer, 
        milestones, 
        gamma=0.1,
        last_epoch=-1,
    ):
        super(MultiStepLR, self).__init__(
            optimizer, 
            milestones, 
            gamma,
            last_epoch
        )

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        return {
            "optimizer": optimizer,
            "milestones": [i * data_size for i in cfg.LR_SCHEDULER.MILESTONES],
            "gamma": cfg.LR_SCHEDULER.GAMMA,
            "last_epoch": -1
        }

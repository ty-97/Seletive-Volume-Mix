"""
hacked together by Zhaofan Qiu, Copyright 2022.
"""

import torch
from pytorchvideo.config import configurable
from .build import LR_SCHEDULER_REGISTRY

@LR_SCHEDULER_REGISTRY.register()
class StepLR(torch.optim.lr_scheduler.StepLR):
    @configurable
    def __init__(
        self, 
        *,
        optimizer, 
        step_size, 
        gamma=0.1
    ):
        super(StepLR, self).__init__(
            optimizer, 
            step_size, 
            gamma
        )

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        return {
            "optimizer": optimizer,
            "step_size": cfg.LR_SCHEDULER.STEP_SIZE * data_size,
            "gamma": cfg.LR_SCHEDULER.GAMMA,
        }

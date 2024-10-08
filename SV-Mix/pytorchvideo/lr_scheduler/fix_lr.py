"""
hacked together by Zhaofan Qiu, Copyright 2022.
"""

from torch.optim.lr_scheduler import LambdaLR
from pytorchvideo.config import configurable
from .build import LR_SCHEDULER_REGISTRY

@LR_SCHEDULER_REGISTRY.register()
class FixLR(LambdaLR):
    """ 
    Fix LR
    """
    @configurable
    def __init__(
        self, 
        *,
        optimizer, 
        last_epoch=-1
    ):
        super(FixLR, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        return {
            "optimizer": optimizer,
            "last_epoch": -1
        }

    def lr_lambda(self, step):
        return 1.
"""
by Yan Shu, Copyright 2022.
"""
import torch.nn as nn
from pytorchvideo.config import kfg, configurable
from .build import LOSSES_REGISTRY


@LOSSES_REGISTRY.register()
class MSE(nn.Module):
    """
    MSE loss
    """
    @configurable
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        logits = batched_inputs[kfg.LOGITS]
        logits_t = batched_inputs[kfg.LOGITS_T]
        return self._forward(logits, logits_t)

    def _forward(self, pred, target):
        loss = self.criterion(pred, target)
        return {"mse_loss": loss}

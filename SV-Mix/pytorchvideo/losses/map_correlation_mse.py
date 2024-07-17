"""
by Yan Shu, copyright 2022.
"""
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.config import configurable
from pytorchvideo.config import kfg
from .build import LOSSES_REGISTRY


@LOSSES_REGISTRY.register()
class MapCorrelationMSE(nn.Module):
    """
    Feature map correlation loss

    Args:
        reduction (str, optional): specifies reduction to apply to the output. It can be
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``
    """
    @configurable
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_input):
        x = batched_input[kfg.MAP]  # B L C
        y = batched_input[kfg.MAP_T]  # B L C

        corr_x = x @ x.transpose(-2, -1)
        corr_y = y @ y.transpose(-2, -1)
        corr_x = corr_x.flatten(1)
        corr_y = corr_y.flatten(1)
        corr_x = F.normalize(corr_x, p=2, dim=1)
        corr_y = F.normalize(corr_y, p=2, dim=1)

        loss = F.mse_loss(corr_x, corr_y, reduction=self.reduction)
        return {'map_correlation_mse_loss': loss}

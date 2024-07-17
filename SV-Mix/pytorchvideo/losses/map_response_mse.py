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
class MapResponseMSE(nn.Module):
    """
    Feature map response loss

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

    def forward(self, batched_inputs):
        x = batched_inputs[kfg.MAP]  # B L C
        y = batched_inputs[kfg.MAP_T]  # B L C

        x = torch.linalg.norm(x, 1, dim=-1)
        x = F.normalize(x, dim=1)
        y = torch.linalg.norm(y, 1, dim=-1)
        y = F.normalize(y, dim=1)

        loss = F.mse_loss(x, y, reduction=self.reduction)
        return {'map_response_mse_loss': loss}

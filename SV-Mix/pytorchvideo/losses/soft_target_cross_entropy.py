"""
by Zhaofan Qiu, Copyright 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.config import configurable
from pytorchvideo.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    @configurable
    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        x = batched_inputs[kfg.LOGITS]
        y = batched_inputs[kfg.LABELS]
        if isinstance(x, list):
            dict = {}
            for idx, x_i in enumerate(x):
                dict["soft_target_cross_entropy_loss" + str(idx)] = self._forward(x_i, y)
            return dict
        else:
            return { "soft_target_cross_entropy_loss": self._forward(x, y) }

    def _forward(self, x, y):
        loss = torch.sum(-y * nn.functional.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "none":
            pass
        else:
            raise NotImplementedError
        return loss
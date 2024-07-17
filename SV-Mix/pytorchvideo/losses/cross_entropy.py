"""
by Zhaofan Qiu, Copyright 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.config import configurable
from pytorchvideo.config import kfg
from pytorchvideo.config import CfgNode as CN
from .build import LOSSES_REGISTRY


@LOSSES_REGISTRY.register()
class CrossEntropy(nn.Module):
    @configurable
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        logits = batched_inputs[kfg.LOGITS]
        target = batched_inputs[kfg.LABELS]
        if isinstance(logits, list):
            dict = {}
            for idx, logits_i in enumerate(logits):
                dict["cross_entropy_loss" + str(idx)] = self._forward(logits_i, target)
            return dict
        else:
            return { "cross_entropy_loss": self._forward(logits, target) }

    def _forward(self, pred, target):
        loss = self.criterion(pred, target)
        return loss
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
class LabelSmoothing(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        smoothing_ratio
    ):
        super(LabelSmoothing, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    @classmethod
    def from_config(cls, cfg):
        return {
            "smoothing_ratio": cfg.LOSSES.LABELSMOOTHING
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        logits = batched_inputs[kfg.LOGITS]
        target = batched_inputs[kfg.LABELS]
        if isinstance(logits, list):
            dict = {}
            for idx, logits_i in enumerate(logits):
                dict["label_smoothing_loss" + str(idx)] = self._forward(logits_i, target)
            return dict
        else:
            return { "label_smoothing_loss": self._forward(logits, target) }

    def _forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss
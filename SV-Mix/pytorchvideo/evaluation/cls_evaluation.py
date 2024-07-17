"""
by Zhaofan Qiu, Copyright 2022.
"""

import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import torch

from pytorchvideo.config import kfg, configurable
from pytorchvideo.utils.comm import all_gather, is_main_process, synchronize
from pytorchvideo.utils.file_io import PathManager
from .evaluator import DatasetEvaluator, AverageMeter, accuracy
from pytorchvideo.utils import comm
from .build import EVALUATION_REGISTRY

__all__ = ["ClsEvaluator"]

@EVALUATION_REGISTRY.register()
class ClsEvaluator(DatasetEvaluator):
    @configurable
    def __init__(self):
        super(ClsEvaluator, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    def evaluate(self, cfg, model, test_data_loader, epoch):
        eval_res = {}
        preds = []
        targets = []
        for idx, data in enumerate(test_data_loader):
            data = comm.unwrap_model(model).preprocess_batch(data)
            # forward
            with torch.cuda.amp.autocast(enabled=cfg.ENGINE.FP16):
                pred = model(data)
            target = data[kfg.LABELS].view(-1)
            preds.append(pred)
            targets.append(target)
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)
        if comm.get_world_size() > 1:
            comm.synchronize()
            preds_list = comm.all_gather(preds.cpu())
            targets_list = comm.all_gather(targets.cpu())
            preds_list = [torch.from_numpy(ent.cpu().data.numpy()) for ent in preds_list]
            targets_list = [torch.from_numpy(ent.cpu().data.numpy()) for ent in targets_list]
            preds = torch.cat(preds_list, 0)
            targets = torch.cat(targets_list, 0)
        # preds: [B x num_clips x num_crops, C]
        N, C = preds.shape
        preds = preds.reshape([N // cfg.DATALOADER.NUM_CLIPS // cfg.DATALOADER.NUM_CROPS,
                               cfg.DATALOADER.NUM_CLIPS,
                               cfg.DATALOADER.NUM_CROPS,
                               C]
                              )
        targets = targets.reshape([N // cfg.DATALOADER.NUM_CLIPS // cfg.DATALOADER.NUM_CROPS,
                               cfg.DATALOADER.NUM_CLIPS * cfg.DATALOADER.NUM_CROPS]
                              )
        targets = targets[:, 0]

        preds_single_crop = preds[:, :, 0].mean(dim=1)
        res = accuracy(preds_single_crop, targets, topk=(1, 5))
        eval_res['ACC-1crop@1'] = res[0].item()
        eval_res['ACC-1crop@5'] = res[1].item()
        if cfg.DATALOADER.NUM_CROPS == 3:
            preds_three_crop = preds.mean(dim=1).mean(dim=1)
            res = accuracy(preds_three_crop, targets, topk=(1, 5))
            eval_res['ACC-3crop@1'] = res[0].item()
            eval_res['ACC-3crop@5'] = res[1].item()
        return eval_res
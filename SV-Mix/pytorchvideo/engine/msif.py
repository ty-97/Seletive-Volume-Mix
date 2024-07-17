# -*- coding: utf-8 -*-
"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py
Copyright (c) Facebook, Inc. and its affiliates.
hacked together by Zhaofan Qiu, Copyright 2022.
"""
import time
import logging
import numpy as np
from typing import Dict

import torch

from pytorchvideo.config import kfg
from pytorchvideo.utils import comm
from pytorchvideo.utils.events import get_event_storage
from pytorchvideo.model import build_model

from .build import ENGINE_REGISTRY
from .defaults import create_ddp_model, DefaultTrainer

"""
This file contains ENGINE for training MSImageFormer.
"""

__all__ = [
    "MSImageFormerTrainer",
]


@ENGINE_REGISTRY.register()
class MSImageFormerTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.MODEL.ARCH_CFGS[0] == 'w_mel':
            kfg.LOGITS_T = 'LOGITS_T'
            kfg.MAP = 'MAP'
            kfg.MAP_T = 'MAP_T'

            config_file_t = cfg.MODEL.ARCH_CFGS[5]
            cfg_t = cfg.clone()
            cfg_t.merge_from_file(config_file_t)

            model_t = self.build_teacher(cfg_t)
            self.model_t = create_ddp_model(model_t, broadcast_buffers=False)
            self.model_t.eval()

            self.losses_t = self.build_losses(cfg_t)

    @classmethod
    def build_teacher(cls, cfg_t):
        model_t = build_model(cfg_t)
        logger = logging.getLogger(__name__)
        logger.info("Model_t:\n{}".format(model_t))
        return model_t

    def _write_metrics(
            self,
            loss_dict: Dict[str, torch.Tensor],
            data_time: float,
            prefix: str = "",
            weight_dict=None
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            loss_t_dict (dict): dict of scalar losses of teacher model
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix to add to storage keys
        """
        metrics_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics_dict.update({k: v.detach().cpu().item()})
            else:
                metrics_dict.update({k: v})
        # metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum([metrics_dict[k] * weight_dict[k]
                                        for k in metrics_dict if not str(k).endswith('_t')])
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        logger = logging.getLogger(__name__)

        start = time.perf_counter()

        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            if comm.get_world_size() > 1:
                self.train_data_loader.sampler.set_epoch(self.iter // self.iters_per_epoch)
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)

            if not self.cfg.DATALOADER.SHUFFLE:
                self.train_data_loader.dataset._map_func._obj.epoch = self.iter // self.iters_per_epoch

        data_time = time.perf_counter() - start
        data = comm.unwrap_model(self.model).preprocess_batch(data)

        if self.mixup:
            x, label = self.mixup(data[kfg.FRAMES], data[kfg.LABELS])
            data.update({
                kfg.FRAMES: x,
                kfg.LABELS: label
            })

        with torch.cuda.amp.autocast(enabled=self.cfg.ENGINE.FP16):
            if self.cfg.MODEL.ARCH_CFGS[0] == 'w_mel':
                feature = self.model(data, True)
                with torch.no_grad():
                    feature_t = self.model_t(data, True)
                data.update({
                    kfg.MAP: feature[0],
                    kfg.LOGITS: feature[1],
                    kfg.MAP_T: feature_t[0],
                    kfg.LOGITS_T: feature_t[1]
                })
            else:
                logits = self.model(data)
                data.update({kfg.LOGITS: logits})
            losses_dict = {}
            weight_dict = {}
            for loss_idx, loss in enumerate(self.losses):
                loss_dict = loss(data)
                losses_dict.update(loss_dict)
                weight_dict.update({list(loss_dict.keys())[0]: self.cfg.LOSSES.WEIGHTS[loss_idx]})

            losses_t_dict = {}
            if self.cfg.MODEL.ARCH_CFGS[0] == 'w_mel':
                data_t = {
                    kfg.LOGITS: data[kfg.LOGITS_T],
                    kfg.LABELS: data[kfg.LABELS]
                }
                for loss in self.losses_t:
                    loss_t_dict = loss(data_t)
                    losses_t_dict.update({k + '_t': loss_t_dict[k] for k in loss_t_dict})

        losses = [losses_dict[k] * weight_dict[k] for k in losses_dict if 'acc' not in k]
        losses = sum(losses)
        losses_dict.update(losses_t_dict)
        self._write_metrics(losses_dict, data_time, weight_dict=weight_dict)

        self.loss_scaler.backward(losses)

        if self.iter % self.iter_size == 0:
            self.loss_scaler.update(self.optimizer)

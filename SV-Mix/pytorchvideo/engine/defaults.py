# -*- coding: utf-8 -*-
"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py
Copyright (c) Facebook, Inc. and its affiliates.
hacked together by Zhaofan Qiu, Copyright 2022.
"""

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""
import time
import logging
import tqdm
import os
import itertools
import numpy as np
import weakref
from collections import OrderedDict
from typing import Dict, List, Optional
from omegaconf import OmegaConf

import torch
from torch.nn.parallel import DistributedDataParallel

from pytorchvideo.config import kfg
from pytorchvideo.utils import comm
from pytorchvideo.utils.collect_env import collect_env_info
from pytorchvideo.utils.env import TORCH_VERSION, seed_all_rng
from pytorchvideo.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, get_event_storage
from pytorchvideo.utils.file_io import PathManager
from pytorchvideo.utils.logger import setup_logger

from pytorchvideo.checkpoint import PtCheckpointer
from pytorchvideo.datasets import build_video_train_loader, build_video_test_loader
from pytorchvideo.model import build_model
from pytorchvideo.optim import build_optimizer
from pytorchvideo.lr_scheduler import build_lr_scheduler
from pytorchvideo.losses import build_losses
from pytorchvideo.evaluation import build_evaluation
from pytorchvideo.utils.cuda import NativeScaler
from pytorchvideo.transform.mixup import MixUp
from pytorchvideo.transform.maskmix import MaskMix
from pytorchvideo.transform.motioncut import MotionCut
from pytorchvideo.transform.motioncut_static import MotionCutStatic
from pytorchvideo.transform.saliencycut import SaliencyCut
from pytorchvideo.transform.maskmotioncut import MaskMotionCut
from pytorchvideo.transform.maskmotioncut_static import MaskMotionCutStatic

from . import hooks
from .train_loop import TrainerBase
from .build import ENGINE_REGISTRY

__all__ = [
    "default_setup",
    "default_writers",
    "create_ddp_model",
    "DefaultTrainer",
]


def default_setup(cfg, args):
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    #if not (hasattr(args, "eval_only") and args.eval_only):
    #    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    return [
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, find_unused_parameters=True, **kwargs)
    #if fp16_compression:
    #    from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
    #    ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp    


@ENGINE_REGISTRY.register()
class DefaultTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        logger = logging.getLogger("pytorchvideo")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        
        if cfg.TRANSFORM.MIXUP_PROB == 0.0:
            self.mixup = None
        else:
            assert cfg.TRANSFORM.MASK_MIX + cfg.TRANSFORM.MOTION_CUT + cfg.TRANSFORM.SALIENCY_CUT + \
                    cfg.TRANSFORM.MASK_MOTION_CUT + cfg.TRANSFORM.MOTION_CUT_STATIC + cfg.TRANSFORM.MASK_MOTION_CUT_STATIC<= 1 \
                and cfg.TRANSFORM.MASK_MIX in [0,1] and cfg.TRANSFORM.MOTION_CUT in [0,1] and cfg.TRANSFORM.SALIENCY_CUT in [0,1]\
                     and cfg.TRANSFORM.MASK_MOTION_CUT in [0,1] and cfg.TRANSFORM.MOTION_CUT_STATIC in [0,1] and cfg.TRANSFORM.MASK_MOTION_CUT_STATIC in [0,1]
            if cfg.TRANSFORM.MASK_MIX > 0:
                self.mixup = MaskMix(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
                                     switch_prob=cfg.TRANSFORM.SWITCH_PROB,
                                     label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
                                     num_classes=cfg.MODEL.NUM_CLASSES)
            elif cfg.TRANSFORM.MOTION_CUT > 0:
                self.mixup = MotionCut(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
                                       switch_prob=cfg.TRANSFORM.SWITCH_PROB,
                                       label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
                                       num_classes=cfg.MODEL.NUM_CLASSES)
            elif cfg.TRANSFORM.MOTION_CUT_STATIC > 0:
                self.mixup = MotionCutStatic(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
                                       switch_prob=cfg.TRANSFORM.SWITCH_PROB,
                                       label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
                                       num_classes=cfg.MODEL.NUM_CLASSES)
            elif cfg.TRANSFORM.SALIENCY_CUT > 0:
                self.mixup = SaliencyCut(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
                                         switch_prob=cfg.TRANSFORM.SWITCH_PROB,
                                         label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
                                         num_classes=cfg.MODEL.NUM_CLASSES)
            elif cfg.TRANSFORM.MASK_MOTION_CUT > 0:
                self.mixup = MaskMotionCut(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
                                         switch_prob=cfg.TRANSFORM.SWITCH_PROB,
                                         label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
                                         num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
            elif cfg.TRANSFORM.MASK_MOTION_CUT_STATIC > 0:
                self.mixup = MaskMotionCutStatic(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
                                         switch_prob=cfg.TRANSFORM.SWITCH_PROB,
                                         label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
                                         num_classes=cfg.MODEL.NUM_CLASSES)
            else:
                self.mixup = MixUp(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
                                   switch_prob=cfg.TRANSFORM.SWITCH_PROB,
                                   label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
                                   num_classes=cfg.MODEL.NUM_CLASSES)


        model = self.build_model(cfg)
        opt, opt_mixblock = self.build_optimizer(cfg, model, self.mixup.mix_block_s)
        self.optimizer = opt
        self.optimizer_mixblock = opt_mixblock
        self.train_data_loader = self.build_train_loader(cfg)
        self.test_data_loader = self.build_test_loader(cfg)
        self.iters_per_epoch = len(self.train_data_loader)
        self._train_data_loader_iter = iter(self.train_data_loader)
        self.iter_size = cfg.DATALOADER.TOTAL_BATCH_SIZE // cfg.DATALOADER.TRAIN_BATCH_SIZE // comm.get_world_size()
        
        self.losses = self.build_losses(cfg)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, self.iters_per_epoch)
        self.scheduler_mixblock = self.build_lr_scheduler(cfg, self.optimizer_mixblock, self.iters_per_epoch)
        self.evaluator = build_evaluation(cfg) if self.test_data_loader is not None else None

        
        self.model = create_ddp_model(model, broadcast_buffers=False)
        self.model = torch.compile(self.model)
        self.model.train()

        self.optimizer.zero_grad()
        self.optimizer_mixblock.zero_grad()
        
        self.checkpointer = PtCheckpointer(
            comm.unwrap_model(self.model),
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
            mix_bloc_s=comm.unwrap_model(self.mixup.mix_block_s),
            model_ofl=comm.unwrap_model(self.mixup.tsm_model),
        )

        # if cfg.TRANSFORM.MIXUP_PROB == 0.0:
        #     self.mixup = None
        # else:
        #     assert cfg.TRANSFORM.MASK_MIX + cfg.TRANSFORM.MOTION_CUT + cfg.TRANSFORM.SALIENCY_CUT + \
        #             cfg.TRANSFORM.MASK_MOTION_CUT + cfg.TRANSFORM.MOTION_CUT_STATIC + cfg.TRANSFORM.MASK_MOTION_CUT_STATIC<= 1 \
        #         and cfg.TRANSFORM.MASK_MIX in [0,1] and cfg.TRANSFORM.MOTION_CUT in [0,1] and cfg.TRANSFORM.SALIENCY_CUT in [0,1]\
        #              and cfg.TRANSFORM.MASK_MOTION_CUT in [0,1] and cfg.TRANSFORM.MOTION_CUT_STATIC in [0,1] and cfg.TRANSFORM.MASK_MOTION_CUT_STATIC in [0,1]
        #     if cfg.TRANSFORM.MASK_MIX > 0:
        #         self.mixup = MaskMix(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
        #                              switch_prob=cfg.TRANSFORM.SWITCH_PROB,
        #                              label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
        #                              num_classes=cfg.MODEL.NUM_CLASSES)
        #     elif cfg.TRANSFORM.MOTION_CUT > 0:
        #         self.mixup = MotionCut(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
        #                                switch_prob=cfg.TRANSFORM.SWITCH_PROB,
        #                                label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
        #                                num_classes=cfg.MODEL.NUM_CLASSES)
        #     elif cfg.TRANSFORM.MOTION_CUT_STATIC > 0:
        #         self.mixup = MotionCutStatic(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
        #                                switch_prob=cfg.TRANSFORM.SWITCH_PROB,
        #                                label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
        #                                num_classes=cfg.MODEL.NUM_CLASSES)
        #     elif cfg.TRANSFORM.SALIENCY_CUT > 0:
        #         self.mixup = SaliencyCut(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
        #                                  switch_prob=cfg.TRANSFORM.SWITCH_PROB,
        #                                  label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
        #                                  num_classes=cfg.MODEL.NUM_CLASSES)
        #     elif cfg.TRANSFORM.MASK_MOTION_CUT > 0:
        #         self.mixup = MaskMotionCut(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
        #                                  switch_prob=cfg.TRANSFORM.SWITCH_PROB,
        #                                  label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
        #                                  num_classes=cfg.MODEL.NUM_CLASSES, cfg=cfg)
        #     elif cfg.TRANSFORM.MASK_MOTION_CUT_STATIC > 0:
        #         self.mixup = MaskMotionCutStatic(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
        #                                  switch_prob=cfg.TRANSFORM.SWITCH_PROB,
        #                                  label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
        #                                  num_classes=cfg.MODEL.NUM_CLASSES)
        #     else:
        #         self.mixup = MixUp(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=cfg.TRANSFORM.MIXUP_PROB,
        #                            switch_prob=cfg.TRANSFORM.SWITCH_PROB,
        #                            label_smoothing=cfg.TRANSFORM.MIXUP_LABELSMOOTHING,
        #                            num_classes=cfg.MODEL.NUM_CLASSES)


        self.loss_scaler = NativeScaler()

        self.best_score = 0
        self.cfg = cfg
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.EPOCH * self.iters_per_epoch
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        self.checkpointer.resume_or_load(self.cfg.SOLVER.SNAPSHOT, resume=resume)
        #print(self.cfg.SOLVER.SNAPSHOT,'\n\n\n\n\n')
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = self.iter + 1

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
        ]

        def test_and_save_results(epoch):
            self._last_eval_results = self.test(self.cfg, self.mixup.tsm_model, self.test_data_loader, self.evaluator, epoch)
            score = sum([self._last_eval_results[k] for k in self._last_eval_results if k in self.cfg.ENGINE.BEST_METRICS])
            if self.best_score < score:
                self.best_score = score
            return self._last_eval_results

        if self.test_data_loader is not None:
            ret.append(hooks.EvalHook(cfg.SOLVER.EVAL_PERIOD, cfg.SOLVER.DENSE_EVAL_EPOCH, test_and_save_results, self.iters_per_epoch))

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD * self.iters_per_epoch))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.SOLVER.WRITE_PERIOD))
        return ret

    def build_writers(self):
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        super().train(self.start_iter, self.max_iter)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model, mix_block_s=None):
        return build_optimizer(cfg, model, mix_block_s)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, iters_per_epoch):
        return build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_video_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        return build_video_test_loader(cfg)

    @classmethod
    def build_losses(cls, cfg):
        return build_losses(cfg)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        ret["optimizer_mixblock"] = self.optimizer_mixblock.state_dict()
        ret["best_score"] = self.best_score
        ret["scheduler"] = self.scheduler.state_dict()
        ret["scheduler_mixblock"] = self.scheduler_mixblock.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])

        self.optimizer_mixblock.load_state_dict(state_dict["optimizer_mixblock"])
        self.scheduler_mixblock.load_state_dict(state_dict["scheduler_mixblock"])

        self.best_score = state_dict["best_score"]

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix to add to storage keys
        """
        metrics_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics_dict.update({k: v.detach().cpu().item()})
            else:
                metrics_dict.update({k: v})
        #metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
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
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    @classmethod
    def test(cls, cfg, model, test_data_loader, evaluator, epoch):
        model.eval()
        with torch.no_grad():
            eval_res = evaluator.evaluate(cfg, model, test_data_loader, epoch)
        model.train()
        # try:
        #     model.module.drop.eval()
        # except AttributeError:
        #     model.drop.eval()
        for n,m in model.named_modules():
            if 'drop' in n:
                m.eval()
                print(n)
        return eval_res

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        logger = logging.getLogger(__name__)

        start = time.perf_counter()
        
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            if comm.get_world_size() > 1:
                self.train_data_loader.sampler.set_epoch(self.iter//self.iters_per_epoch)     
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)

            if not self.cfg.DATALOADER.SHUFFLE:
                self.train_data_loader.dataset._map_func._obj.epoch = self.iter//self.iters_per_epoch

        data_time = time.perf_counter() - start
        data = comm.unwrap_model(self.model).preprocess_batch(data)

        if self.mixup:
            x, label, x_ofl, label_ofl, mask_loss = self.mixup(data[kfg.FRAMES], data[kfg.LABELS])
            data.update({
                kfg.FRAMES: x,
                kfg.LABELS: label
            })
            data_ofl = {kfg.FRAMES: x_ofl,
                kfg.LABELS: label_ofl}
            
        with torch.cuda.amp.autocast(enabled=self.cfg.ENGINE.FP16):
            logits = self.model(data)
            #print(logits.shape)
            logits_ofl = self.mixup.tsm_model(data_ofl)
            #print(logits_ofl.requires_grad)
            data.update({ kfg.LOGITS: logits })
            data_ofl.update({ kfg.LOGITS: logits_ofl })
            losses_dict = {}
            weight_dict = {}

            losses_dict_ofl = {}
            weight_dict_ofl = {}

            for loss_idx, loss in enumerate(self.losses):
                loss_dict = loss(data)
                loss_dict_ofl = loss(data_ofl)
                losses_dict.update(loss_dict)
                losses_dict_ofl.update(loss_dict_ofl)

                for key_i in range(len(list(loss_dict.keys()))):
                    weight_dict.update({list(loss_dict.keys())[key_i]: self.cfg.LOSSES.WEIGHTS[loss_idx]})
                for key_i in range(len(list(loss_dict_ofl.keys()))):
                    weight_dict_ofl.update({list(loss_dict_ofl.keys())[key_i]: self.cfg.LOSSES.WEIGHTS[loss_idx]})

        losses = [losses_dict[k] * weight_dict[k] for k in losses_dict if 'acc' not in k]
        losses = sum(losses)

        losses_ofl = [losses_dict_ofl[k] * weight_dict_ofl[k] for k in losses_dict_ofl if 'acc' not in k]
        losses_ofl = sum(losses_ofl)

        #print(losses,'\n',losses_ofl)

        losses = losses + losses_ofl + mask_loss

        #print(losses,losses_ofl,mask_loss)
        losses_dict.update({'loss_ofl':losses_ofl, 'mask_loss':mask_loss})
        self._write_metrics(losses_dict, data_time)
        #self._write_metrics(losses_dict_ofl, data_time)

        #print(self.model.conv1.weight[0,0,0,0:5])
        self.loss_scaler.backward(losses / self.iter_size)
        #print(self.model.layer1[0].conv1.weight[0:5,0,0,0],self.model.layer1[0].conv1.weight.grad[0:5,0,0,0])
        if (self.iter + 1) % self.iter_size == 0:
            self.loss_scaler.update(self.optimizer)
            self.loss_scaler.update(self.optimizer_mixblock)
        #print(self.model.layer1[0].conv1.weight[0:5,0,0,0],self.model.layer1[0].conv1.weight.grad[0:5,0,0,0])
        #print(self.model.conv1.weight[0,0,0,0:5],self.model.conv1.weight.grad[0,0,0,0:5],self.model.conv1.weight.grad.shape,self.model.conv1.weight.shape)
        # print(self.mixup.mix_block.value.weight[0,0:5,0,0],self.mixup.mix_block.value.weight.grad[0,0:5,0,0])

"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/checkpoint/detection_checkpoint.py
Copyright (c) Facebook, Inc. and its affiliates.
hacked together by Zhaofan Qiu, Copyright 2022.
"""

import logging
import os
import pickle
import torch
from typing import Any
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer, _IncompatibleKeys
from fvcore.common.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from torch.nn.parallel import DistributedDataParallel

import pytorchvideo.utils.comm as comm
from pytorchvideo.utils.file_io import PathManager


class PeriodicEpochCheckpointer(PeriodicCheckpointer):
    def step(self, iteration: int, epoch: int, best_score: float, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        epoch = int(epoch)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if epoch == 0:
            self.score = 0

        if (iteration + 1) % self.period == 0:
            #self.checkpointer.save(
            #    "{}_Epoch_{:05d}_Iter_{:07d}".format(self.file_prefix, epoch, iteration), **additional_state
            #)
            self.checkpointer.save('current', **additional_state)
            if best_score > self.score:
                self.score = best_score
                self.checkpointer.save('best', **additional_state)

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                # pyre-fixme[58]: `>` is not supported for operand types `int` and
                #  `Optional[int]`.
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(
                        file_to_delete
                    ) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        self.path_manager.rm(file_to_delete)

        if self.max_iter is not None:
            # pyre-fixme[58]
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)


class PtCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in pytorchvideo
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.path_manager = PathManager

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        incompatible = super()._load_model(checkpoint)
        if incompatible is None:  # support older versions of fvcore
            return None

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible

    def _log_incompatible_keys(self, incompatible: _IncompatibleKeys) -> None:
        """
        Log information about the incompatible keys returned by ``_load_model``.
        """
        for k, shape_checkpoint, shape_model in incompatible.incorrect_shapes:
            self.logger.warning(
                "Skip loading parameter '{}' to the model due to incompatible "
                "shapes: {} in the checkpoint but {} in the "
                "model! You might want to double check if this is expected.".format(
                    k, shape_checkpoint, shape_model
                )
            )
        if incompatible.missing_keys:
            self.logger.info(get_missing_parameters_message(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            self.logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

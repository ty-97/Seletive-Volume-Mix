# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import inspect
import logging

import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from pytorchvideo.config import kfg
from pytorchvideo.utils import comm
from pytorchvideo.utils.func_tensor import dict_to_cuda
from pytorchvideo.layers.frozen_bn import FrozenBatchNorm


class BaseNet(nn.Module, metaclass=ABCMeta):
    _arch_dict = {}

    def __init__(self):
        super(BaseNet, self).__init__()

    @abstractmethod
    def forward(self, batched_inputs):
        pass

    @classmethod
    def from_config(cls, cfg) -> dict:
        arch_name = cfg.MODEL.ARCH
        arch_cfgs = cls._arch_dict.get(arch_name)
        if arch_cfgs is None:
            raise AttributeError(f'Unsupported ARCH type {arch_name} for {cfg.MODEL.NAME}')
        arch_cfgs = cls.merge_arch_cfg(**arch_cfgs)

        input_arch_cfgs = {
            'num_classes': cfg.MODEL.NUM_CLASSES,
            'weights': cfg.MODEL.WEIGHTS,
            'transfer_weights': cfg.MODEL.TRANSFER_WEIGHTS,
            'remove_fc': cfg.MODEL.REMOVE_FC,
            'drop_path_rate': cfg.MODEL.DROP_PATH_RATIO
            
        }

        # Update arch_cfg with input_cfg
        # Will overwrite existing keys in arch_cfg
        arch_cfgs.update(input_arch_cfgs)
        return arch_cfgs

    @classmethod
    def merge_arch_cfg(cls, **kwargs) -> dict:
        """
        Merge default arguments of ``cls.__init__`` method with input kwargs

        Args:
            **kwargs: input arguments from ``cls._arch_dict``

        Returns:
            Updated arguments

        """
        default_cfgs = inspect.signature(cls.__init__).parameters
        arch_cfgs = {}
        append_rest = False
        for key, param in default_cfgs.items():
            if key == 'self':
                continue
            if param.kind is param.VAR_KEYWORD:
                append_rest = True
                continue
            arch_cfgs[key] = kwargs.pop(key, param.default)
        if append_rest:
            kwargs.pop('self')
            arch_cfgs.update(kwargs)

        return arch_cfgs

    def load_pretrained(self, weights, transfer_weights, remove_fc, *args, **kwargs):
        """Load pretrained model"""
        logger = logging.getLogger(__name__)

        ckpt = torch.load(weights, map_location='cpu')
        if 'model' in ckpt:
            state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt['model'].items()}
        elif 'state_dict' in ckpt:
            state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt['state_dict'].items()}
        else:
            state_dict = ckpt

        # convert initial weights
        if transfer_weights:
            state_dict = self.transfer_weights(state_dict, *args, **kwargs)
        if remove_fc:
            state_dict = self.remove_fc(state_dict)

        [misskeys, unexpkeys] = self.load_state_dict(state_dict, strict=False)
        logger.info('Missing keys: {}'.format(misskeys))
        logger.info('Unexpect keys: {}'.format(unexpkeys))
        logger.info("==> loaded checkpoint '{}'".format(weights))

    @staticmethod
    def transfer_weights(state_dict, *args, **kwargs):
        """Transfer weights in  ``state_dict``"""
        return state_dict

    @staticmethod
    def remove_fc(state_dict):
        """Remove parameters of fc layers in state_dict"""
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)

        state_dict.pop('fc_dual.weight', None)
        state_dict.pop('fc_dual.bias', None)

        state_dict.pop('fc_dist.weight', None)
        state_dict.pop('fc_dist.bias', None)
        return state_dict

    @classmethod
    def frozen_bn(cls, module, full_name='root'):
        """Frozen the parameters of BN layers"""
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = FrozenBatchNorm(module.num_features,
                                            module.eps, module.momentum,
                                            module.affine,
                                            module.track_running_stats)
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked

        skip_first_bn = (full_name == 'root')
        for name, child in module.named_children():
            if skip_first_bn and isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
                skip_first_bn = False
                if comm.get_rank() == 0:
                    print('skip frozen bn: ' + full_name + '.' + name)
                continue
            module_output.add_module(name, cls.frozen_bn(child, full_name + '.' + name))
        del module
        return module_output

    def preprocess_batch(self, batched_inputs):
        dict_to_cuda(batched_inputs)
        return batched_inputs

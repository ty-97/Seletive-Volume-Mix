"""
by Zhaofan Qiu, Copyright 2022.
"""

import copy
import torch
import itertools
from enum import Enum
from pytorchvideo.layers.frozen_bn import FrozenBatchNorm
from pytorchvideo.config import CfgNode
from pytorchvideo.utils.registry import Registry
import pytorchvideo.utils.comm as comm

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

SOLVER_REGISTRY = Registry("SOLVER")
SOLVER_REGISTRY.__doc__ = """
Registry for SOLVER.
"""

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.SOLVER.GRAD_CLIP, cfg.SOLVER.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.SOLVER.GRAD_CLIP)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        'value': clip_grad_value,
        'norm': clip_grad_norm,
    }
    clipper = _GRADIENT_CLIP_TYPE_TO_CLIPPER[cfg.SOLVER.GRAD_CLIP_TYPE]
    if cfg.SOLVER.GRAD_CLIP_TYPE == 'value':
        return clipper, None
    else:
        return None, clipper


def get_default_optimizer_params(
    model: torch.nn.Module,
    #mix_block_s,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):

    if weight_decay_bias is None:
        weight_decay_bias = weight_decay
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        skip = {}
        if hasattr(module, 'no_weight_decay'):
            skip = module.no_weight_decay()
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            schedule_params = {
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
            if isinstance(module, norm_module_types):
                schedule_params["weight_decay"] = weight_decay_norm
            elif module_param_name in skip:
                schedule_params["weight_decay"] = 0.
            elif module_param_name == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                schedule_params["lr"] = base_lr * bias_lr_factor
                schedule_params["weight_decay"] = weight_decay_bias
            if overrides is not None and module_param_name in overrides:
                schedule_params.update(overrides[module_param_name])
            params += [
                {
                    "params": [value],
                    "lr": schedule_params["lr"],
                    "weight_decay": schedule_params["weight_decay"],
                }
            ]

    # for m in mix_block_s.modules():
    #     if isinstance(m, (torch.nn.Conv1d, 
    #                       torch.nn.Conv2d,
    #                       torch.nn.Conv3d,
    #                       torch.nn.Linear,
    #                       torch.nn.BatchNorm2d,
    #                       torch.nn.BatchNorm3d)):
    #         ps = list(m.parameters())
    #         mix_block_s_weight.append(ps[0])
    #         if len(ps) == 2:
    #             mix_block_s_weight.append(ps[1])
    
    #     elif len(m._modules) == 0:
    #         if len(list(m.parameters())) > 0:
    #             raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
    # params += [{'params': mix_block_s_weight, 'lr': 10*base_lr, 'weight_decay': weight_decay},]
    return params


def get_tsm_optimizer_params(
        model: torch.nn.Module,
        #mix_block_s,
        base_lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        fc_lr5: Optional[bool] = None,
        use_flow: Optional[bool] = None,
        frozen_bn: Optional[bool] = None,
        switch_prob = 0.5
):
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    lr5_weight = []
    lr10_bias = []
    bn = []
    custom_ops = []
    mix_block_s_weight = []

    conv_cnt = 0
    bn_cnt = 0

    for m in model.modules():
        if isinstance(m, FrozenBatchNorm):
            continue
        elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            if fc_lr5:
                lr5_weight.append(ps[0])
            else:
                normal_weight.append(ps[0])
            if len(ps) == 2:
                if fc_lr5:
                    lr10_bias.append(ps[1])
                else:
                    normal_bias.append(ps[1])

        elif isinstance(m, torch.nn.BatchNorm2d):
            bn_cnt += 1
            # later BN's are frozen
            if not frozen_bn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not frozen_bn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    
    # for m in mix_block_s.modules():
    #     if isinstance(m, (torch.nn.Conv1d, 
    #                       torch.nn.Conv2d,
    #                       torch.nn.Conv3d,
    #                       torch.nn.Linear,
    #                       torch.nn.BatchNorm2d,
    #                       torch.nn.BatchNorm3d)):
    #         ps = list(m.parameters())
    #         mix_block_s_weight.append(ps[0])
    #         if len(ps) == 2:
    #             mix_block_s_weight.append(ps[1])
    
    #     elif len(m._modules) == 0:
    #         if len(list(m.parameters())) > 0:
    #             raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    # print({'params': len(first_conv_weight), 'lr': 5 * base_lr if use_flow else base_lr, 'weight_decay': weight_decay},'\n',
    #     {'params': len(first_conv_bias), 'lr': 10 * base_lr if use_flow else 2 * base_lr, 'weight_decay': 0.},'\n',
    #     {'params': len(normal_weight), 'lr': base_lr, 'weight_decay': weight_decay},'\n',
    #     {'params': len(normal_bias), 'lr': 2 * base_lr, 'weight_decay': 0.},'\n',
    #     {'params': len(bn), 'lr': base_lr, 'weight_decay': 0.},'\n',
    #     {'params': len(custom_ops), 'lr': base_lr, 'weight_decay': weight_decay},'\n',
    #     # for fc
    #     {'params': len(lr5_weight), 'lr': 5 * base_lr, 'weight_decay': weight_decay},'\n',
    #     {'params': len(lr10_bias), 'lr': 10 * base_lr, 'weight_decay': 0.},'\n',
    #     )

    return [
        {'params': first_conv_weight, 'lr': 5 * base_lr if use_flow else base_lr, 'weight_decay': weight_decay},
        {'params': first_conv_bias, 'lr': 10 * base_lr if use_flow else 2 * base_lr, 'weight_decay': 0.},
        {'params': normal_weight, 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': normal_bias, 'lr': 2 * base_lr, 'weight_decay': 0.},
        {'params': bn, 'lr': base_lr, 'weight_decay': 0.},
        {'params': custom_ops, 'lr': base_lr, 'weight_decay': weight_decay},
        # for fc
        {'params': lr5_weight, 'lr': 5 * base_lr, 'weight_decay': weight_decay},
        {'params': lr10_bias, 'lr': 10 * base_lr, 'weight_decay': 0.},
        # {'params': mix_block_s_weight, 'lr': (5/switch_prob) * base_lr if switch_prob!=0 else 0, 'weight_decay': weight_decay},
        # {'params': mix_block_t_weight, 'lr': (5/(1-switch_prob)) * base_lr if 1-switch_prob!=0 else 0, 'weight_decay': weight_decay},
        #{'params': mix_block_s_weight, 'lr': base_lr, 'weight_decay': weight_decay},
    ]



def get_mixblock_optimizer_params(
    mix_block_s,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):
    mix_block_s_weight = []
    if weight_decay_bias is None:
        weight_decay_bias = weight_decay
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    for m in mix_block_s.modules():
        if isinstance(m, (torch.nn.Conv1d, 
                          torch.nn.Conv2d,
                          torch.nn.Conv3d,
                          torch.nn.Linear,
                          torch.nn.BatchNorm2d,
                          torch.nn.BatchNorm3d)):
            ps = list(m.parameters())
            mix_block_s_weight.append(ps[0])
            if len(ps) == 2:
                mix_block_s_weight.append(ps[1])
    
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
    params += [{'params': mix_block_s_weight, 'lr': base_lr, 'weight_decay': weight_decay},]
    #print(params[-1]['lr'])
    #xxx
    return params



def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert (
        per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
    cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]
) -> Type[torch.optim.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if cfg.SOLVER.GRAD_CLIP <= 0:
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    per_param_clipper, global_clipper = _create_gradient_clipper(cfg)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=per_param_clipper, global_clipper=global_clipper
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip


def build_optimizer(cfg: CfgNode, model: torch.nn.Module, mix_block_s) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    lr = cfg.SOLVER.BASE_LR
    lr_mixblock = cfg.SOLVER.BASE_LR_MIXBLOCK
    if cfg.SOLVER.AUTO_LR:
        # word_size = comm.get_world_size()
        # lr = cfg.DATALOADER.TRAIN_BATCH_SIZE * word_size / 16 * lr
        lr = cfg.DATALOADER.TOTAL_BATCH_SIZE / 192 * lr

    if cfg.SOLVER.PARAMS == 'default':
        params = get_default_optimizer_params(
            model,
            #mix_block_s,
            base_lr=lr,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )
    elif cfg.SOLVER.PARAMS == 'tsm':
        params = get_tsm_optimizer_params(
            model,
            #mix_block_s,
            base_lr=lr,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            fc_lr5=cfg.SOLVER.FC_LR5,
            use_flow=cfg.TRANSFORM.USE_FLOW,
            frozen_bn=cfg.MODEL.FROZEN_BN,
            switch_prob=cfg.TRANSFORM.SWITCH_PROB 
        )

    params_mixblock = get_mixblock_optimizer_params(
            mix_block_s,
            base_lr=lr_mixblock,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY_MIXBLOCK,
        )


    optimizer = SOLVER_REGISTRY.get(cfg.SOLVER.NAME)
    optimizer_mixblock = SOLVER_REGISTRY.get(cfg.SOLVER.NAME_MIXBLOCK)

    if cfg.SOLVER.GRAD_CLIP_MIXBLOCK != cfg.SOLVER.GRAD_CLIP:
        cfg_mixblock = copy.deepcopy(cfg)
        cfg_mixblock._immutable(False)
        cfg_mixblock.SOLVER.GRAD_CLIP = cfg.SOLVER.GRAD_CLIP_MIXBLOCK
    else:
        cfg_mixblock = cfg
    # print(cfg_mixblock.SOLVER.GRAD_CLIP)
    # print(lr_mixblock)
    #xxx
    return maybe_add_gradient_clipping(cfg, optimizer)(cfg, params), \
        maybe_add_gradient_clipping(cfg_mixblock, optimizer_mixblock)(cfg_mixblock, params_mixblock)


def build_optimizer_params(cfg: CfgNode, params) -> torch.optim.Optimizer:
    optimizer = SOLVER_REGISTRY.get(cfg.SOLVER.NAME)
    return maybe_add_gradient_clipping(cfg, optimizer)(cfg, params)
    

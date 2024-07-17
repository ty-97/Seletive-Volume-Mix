"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/__init__.py
Copyright (c) Facebook, Inc. and its affiliates.
hacked together by Zhaofan Qiu, Copyright 2022.	
"""

import logging
import os
from collections import OrderedDict
import torch
import pytorchvideo.utils.comm as comm
from pytorchvideo.utils.argument_parser import default_argument_parser
from pytorchvideo.checkpoint import PtCheckpointer
from pytorchvideo.config import get_cfg
from pytorchvideo.engine import DefaultTrainer, default_setup, hooks, launch, build_engine


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts) 
    
    if len(cfg.OUTPUT_DIR) == 0:
        basename = os.path.basename(args.config_file)
        basename = os.path.splitext(basename)[0]
        cfg.OUTPUT_DIR = os.path.join(os.path.dirname(args.config_file), 'log_' + basename)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    repeat_time = args.repeat_time

    if args.resume:
        assert repeat_time == 1

    res = []
    for idx in range(repeat_time):
        trainer = build_engine(cfg)
        trainer.resume_or_load(resume=args.resume)

        if args.eval_only and trainer.test_data_loader is not None:
            res = trainer.test(trainer.cfg, trainer.model, trainer.test_data_loader, trainer.evaluator, epoch=-1)
            if comm.is_main_process():
                print(res)
            return res

        trainer.train()
        res.append(trainer.best_score * 100)

    if comm.is_main_process():
        print('best scores: ' + str(res))
        print('mean score: ' + str(sum(res) / repeat_time))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

# -*- coding: utf-8 -*-
"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/checkpoint/__init__.py
Copyright (c) Facebook, Inc. and its affiliates.
hacked together by Zhaofan Qiu, Copyright 2022.
"""

from .ptcheckpoint import PtCheckpointer, PeriodicEpochCheckpointer
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
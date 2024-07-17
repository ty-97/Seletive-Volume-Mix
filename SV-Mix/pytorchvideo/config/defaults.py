"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/config/defaults.py
Copyright (c) Facebook, Inc. and its affiliates.
hacked together by Zhaofan Qiu, Copyright 2022.
"""
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()

_C.DATASET.NAME = ''

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

_C.DATALOADER.TRAIN_BATCH_SIZE = 16

_C.DATALOADER.TEST_BATCH_SIZE = 16

_C.DATALOADER.TOTAL_BATCH_SIZE = 192

_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.ROOT_PATH = ''

_C.DATALOADER.TRAIN_LIST_FILE = ''

_C.DATALOADER.TEST_LIST_FILE = ''

_C.DATALOADER.FORMAT = 'LMDB'

_C.DATALOADER.NUM_CLIPS = 1

_C.DATALOADER.NUM_CROPS = 1

_C.DATALOADER.NUM_SEGMENTS = 1

_C.DATALOADER.CLIP_LENGTH = 1

_C.DATALOADER.NUM_STEPS = 1

_C.DATALOADER.SHUFFLE = True

# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------
_C.ENGINE = CN()

_C.ENGINE.NAME = 'DefaultTrainer'

_C.ENGINE.BEST_METRICS = ['ACC-3crop@1']

_C.ENGINE.FP16 = False

# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
_C.TRANSFORM = CN()

_C.TRANSFORM.NAME = 'DefaultTransform'

_C.TRANSFORM.CROP_SIZE = 224

_C.TRANSFORM.TEST_CROP_SIZE = 256

_C.TRANSFORM.NO_HORIZONTAL_FLIP = False

_C.TRANSFORM.TIME_DIM = 'T'

_C.TRANSFORM.USE_FLOW = False

_C.TRANSFORM.MIXUP_PROB = 0.0

_C.TRANSFORM.MIXUP_LABELSMOOTHING = 0.1

_C.TRANSFORM.SWITCH_PROB = 0.5

_C.TRANSFORM.MASK_MIX = 0.

_C.TRANSFORM.MOTION_CUT = 0.

_C.TRANSFORM.MASK_MOTION_CUT = 0.

_C.TRANSFORM.MASK_MOTION_CUT_STATIC = 0.

_C.TRANSFORM.MOTION_CUT_STATIC = 0.

_C.TRANSFORM.SALIENCY_CUT = 0.

_C.TRANSFORM.AUTO_AUG = "rand-m7-n4-mstd0.5-inc1"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.DEVICE = "cuda"

_C.MODEL.NAME = ''

_C.MODEL.ARCH = ''

_C.MODEL.ARCH_CFGS = []

_C.MODEL.EARLY_STRIDE = 1

_C.MODEL.NUM_CLASSES = 40

_C.MODEL.DROPOUT_RATIO = 0.0

_C.MODEL.DROP_PATH_RATIO = 0.0

_C.MODEL.WEIGHTS = ''

_C.MODEL.TRANSFER_WEIGHTS = False

_C.MODEL.REMOVE_FC = False

_C.MODEL.FROZEN_BN = False

# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
_C.SOLVER = CN()

_C.SOLVER.NAME = 'Adam'

_C.SOLVER.NAME_MIXBLOCK = 'SGD'

_C.SOLVER.PARAMS = 'default'

_C.SOLVER.EPOCH = 10

_C.SOLVER.CHECKPOINT_PERIOD = 1

_C.SOLVER.EVAL_PERIOD = 1

_C.SOLVER.DENSE_EVAL_EPOCH = 1000000

_C.SOLVER.AUTO_LR = False

_C.SOLVER.BASE_LR = 0.0005

_C.SOLVER.BASE_LR_MIXBLOCK = 0.01

_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.FC_LR5 = False

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0

_C.SOLVER.WEIGHT_DECAY_MIXBLOCK = 1e-4

_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.ALPHA = 0.99

_C.SOLVER.BETAS = [0.9, 0.999]

_C.SOLVER.EPS = 1e-8

_C.SOLVER.GRAD_CLIP_TYPE = 'norm' # norm, value

_C.SOLVER.GRAD_CLIP = -1.0 # ignore <= 0

_C.SOLVER.GRAD_CLIP_MIXBLOCK = -1.0 # ignore <= 0

_C.SOLVER.NORM_TYPE = 2.0

_C.SOLVER.WRITE_PERIOD = 20

_C.SOLVER.REGULARIZE = 1.0

_C.SOLVER.SNAPSHOT = ''

# ----------------------------------------------------------------------------
# lr scheduler
# ----------------------------------------------------------------------------
_C.LR_SCHEDULER = CN()

_C.LR_SCHEDULER.NAME = 'StepLR'

_C.LR_SCHEDULER.STEP_SIZE = 3

_C.LR_SCHEDULER.GAMMA = 0.1

_C.LR_SCHEDULER.MILESTONES = (3,)

_C.LR_SCHEDULER.WARMUP_EPOCH = -1

# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #
_C.LOSSES = CN()

_C.LOSSES.NAMES = ['LabelSmoothing']

_C.LOSSES.LABELSMOOTHING = 0.2

_C.LOSSES.WEIGHTS = [1.0]

# ---------------------------------------------------------------------------- #
# Inference
# ---------------------------------------------------------------------------- #
_C.INFERENCE = CN()

_C.INFERENCE.NAME = ''

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""

_C.SEED = -1

_C.CUDNN_BENCHMARK = True

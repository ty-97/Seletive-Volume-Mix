"""
by Zhaofan Qiu, Copyright 2022.
"""

from .build import build_losses

from .cross_entropy import CrossEntropy
from .label_smoothing import LabelSmoothing
from .soft_target_cross_entropy import SoftTargetCrossEntropy
from .mse import MSE
from .map_response_mse import MapResponseMSE
from .map_correlation_mse import MapCorrelationMSE

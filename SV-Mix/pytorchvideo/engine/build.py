"""
by Zhaofan Qiu, Copyright 2022.
"""
from pytorchvideo.utils.registry import Registry

ENGINE_REGISTRY = Registry("ENGINE")
ENGINE_REGISTRY.__doc__ = """
Registry for engine
"""

def build_engine(cfg):
    engine = ENGINE_REGISTRY.get(cfg.ENGINE.NAME)(cfg)
    return engine
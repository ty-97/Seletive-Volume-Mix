"""
Dataset of video classification task
By Zhaofan Qiu, Copyright 2022.
"""
import os
from pytorchvideo.transform.build import build_train_transform, build_test_transform
from .build import DATASETS_REGISTRY
from .data_util import load_list, parse_train_sample, parse_test_sample, load_rgb_frame, load_rgb_lmdb,\
    load_rgb_raw, load_rgb_decord
from pytorchvideo.config import kfg
from pytorchvideo.config import configurable
from pytorchvideo.utils.func_tensor import dict_as_tensor

__all__ = ["VideoClsDataset"]


@DATASETS_REGISTRY.register()
class VideoClsDataset:
    @configurable
    def __init__(self, stage, list_file, root_path, format="LMDB",
                 num_clips=1, num_crops=1,
                 num_segments=1, clip_length=1, num_steps=1,
                 transforms=None):
        super(VideoClsDataset, self).__init__()
        self.stage = stage
        self.list_file = list_file
        self.root_path = root_path
        self.format = format
        self.num_segments = num_segments
        self.clip_length = clip_length
        self.num_steps = num_steps
        self.transform = transforms
        # test-only parameters
        self.num_clips = num_clips
        self.num_crops = num_crops

    @classmethod
    def from_config(cls, cfg, stage):
        ret = {
            "stage": stage,
            "list_file": cfg.DATALOADER.TRAIN_LIST_FILE if stage == 'train' else cfg.DATALOADER.TEST_LIST_FILE,
            "root_path": cfg.DATALOADER.ROOT_PATH,
            "format": cfg.DATALOADER.FORMAT,
            "num_segments": cfg.DATALOADER.NUM_SEGMENTS,
            "clip_length": cfg.DATALOADER.CLIP_LENGTH,
            "num_steps": cfg.DATALOADER.NUM_STEPS,
            "num_clips": cfg.DATALOADER.NUM_CLIPS,
            "num_crops": cfg.DATALOADER.NUM_CROPS,
        }
        if stage == 'train':
            transforms = build_train_transform(cfg)
        else:
            transforms = []
            for i in range(cfg.DATALOADER.NUM_CROPS):
                transforms.append(build_test_transform(cfg, crop_idx=i))
        ret.update({"transforms": transforms})
        return ret

    def load_data(self, cfg):
        samples = load_list(os.path.join(self.root_path, self.list_file))
        if self.stage != 'train':
            samples = [s for s in samples for _ in range(self.num_clips * self.num_crops)]
        return list(enumerate(samples))

    def __call__(self, data):
        data_idx = data[0]
        data_str = data[1]
        if self.stage == 'train':
            video_path, offsets, label = parse_train_sample(data_str, self.num_segments, self.clip_length, self.num_steps)
        else:
            clip_idx = data_idx // self.num_crops % self.num_clips
            video_path, offsets, label = parse_test_sample(data_str, self.num_segments, self.clip_length, self.num_steps, self.num_clips, clip_idx)

        if self.format == "LMDB":
            image_list = load_rgb_lmdb(self.root_path, video_path, offsets, self.num_segments, self.clip_length, self.num_steps)
        elif self.format == "RAW":
            image_list = load_rgb_raw(self.root_path, video_path, offsets, self.num_segments, self.clip_length, self.num_steps)
        elif self.format.startswith("DECORD"):
            image_list = load_rgb_decord(self.root_path, video_path, offsets, self.num_segments, self.clip_length, self.num_steps, self.format)
        elif self.format == "FRAME":
            image_list = load_rgb_frame(self.root_path, video_path, offsets, self.num_segments, self.clip_length, self.num_steps)
        else:
            raise NotImplementedError

        if self.stage == 'train':
            trans_image_list = self.transform(image_list)
        else:
            crop_idx = data_idx % self.num_crops
            trans_image_list = self.transform[crop_idx](image_list)
        ret = {
            kfg.FRAMES: trans_image_list,
            kfg.LABELS: label,
        }
        dict_as_tensor(ret)

        return ret

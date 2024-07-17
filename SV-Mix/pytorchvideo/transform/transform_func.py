"""
Implementation of basic transforms on video data
by Zhaofan Qiu
zhaofanqiu@gmail.com
"""
import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional
import torchvision

import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None

import torchvision.transforms.functional as F
from torchvision import transforms

__all__ = ["ToClipTensor", "ClipRandomResizedCrop", "ClipColorJitter", "ClipRandomGrayscale",
           "ClipRandomHorizontalFlip", "ClipResize", "ClipCenterCrop", "ClipNormalize", "ClipDifference",
           "GroupRandomCrop", "GroupCenterCrop", "GroupRandomHorizontalFlip", "GroupNormalize", "GroupScale",
           "GroupRandomScale", "GroupOverSample", "GroupMultiScaleCrop", "GroupRandomSizedCrop", "Stack",
           "ToTorchFormatTensor", "IdentityTransform"
]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class ToClipTensor(object):
    """Convert a List of ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or numpy.ndarray): Clip to be converted to tensor.

        Returns:
            Tensor: Converted clip.
        """

        return [F.to_tensor(img) for img in clip]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class FlowToClipTensor(ToClipTensor):
    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or numpy.ndarray): Clip of flow to be converted to tensor.

        Returns:
            Tensor: Converted clip.
        """
        clip_len = len(clip)
        tensor_out = []
        for i in range(0, clip_len, 2):
            c1 = F.to_tensor(clip[i])
            c2 = F.to_tensor(clip[i + 1])
            tensor_out.append(torch.cat((c1, c2), dim=0))
        return tensor_out


class ClipRandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be cropped and resized.

        Returns:
            List of PIL Image or Tensor: Randomly cropped and resized clip.
        """
        i, j, h, w = self.get_params(clip[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in clip]


class FlowClipRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, clip):
        """
        Args:
            clip(List of PIL Image or Tensor): Clip of flow to be flipped.

        Returns:
            List of PIL Image or Tensor: Randomly flipped flow.
        """
        if torch.rand(1) < self.p:
            clip_len = len(clip)
            clip_out = []
            for i in range(0, clip_len):
                img = clip[i]
                img = F.hflip(img)
                if i % 2 == 0:
                    img = ImageOps.invert(img)
                clip_out.append(img)
            return clip_out
        else:
            return clip


class ClipColorJitter(transforms.ColorJitter):
     def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Input clip.

        Returns:
            List of PIL Image or Tensor: Color jittered clip.
        """
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                clip = [F.adjust_brightness(img, brightness_factor) for img in clip]

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                clip = [F.adjust_contrast(img, contrast_factor) for img in clip]

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                clip = [F.adjust_saturation(img, saturation_factor) for img in clip]

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                clip = [F.adjust_hue(img, hue_factor) for img in clip]

        return clip


class ClipRandomGrayscale(transforms.RandomGrayscale):
    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be converted to grayscale.

        Returns:
            List of PIL Image or Tensor: Randomly grayscaled clip.
        """
        num_output_channels = 1 if clip[0].mode == 'L' else 3
        if random.random() < self.p:
            return [F.to_grayscale(img, num_output_channels=num_output_channels) for img in clip]
        return clip


class ClipRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be flipped.

        Returns:
            List of PIL Image or Tensor: Randomly flipped clip.
        """
        if torch.rand(1) < self.p:
            return [F.hflip(img) for img in clip]
        return clip


class ClipNormalize(object):
    """Normalize a list of tensor images with mean and standard deviation.
    Given mean: ``(meawam[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): List of tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor list.
        """
        return [F.normalize(img, self.mean, self.std, self.inplace) for img in clip]

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ClipResize(transforms.Resize):
    """Resize the list of PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor):: Clip to be scaled.

        Returns:
            List of PIL Image: Rescaled clip.
        """
        return [F.resize(img, self.size, self.interpolation) for img in clip]


class ClipCenterCrop(transforms.CenterCrop):
    """Crops the given list of PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): Clip to be cropped.

        Returns:
            List of PIL Image: Cropped clip.
        """
        return [F.center_crop(img, self.size) for img in clip]


class ClipFirstCrop(transforms.CenterCrop):
    """Crops the given list of PIL Image at the 1/3.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): Clip to be cropped.

        Returns:
            List of PIL Image: Cropped clip.
        """
        if isinstance(self.size, numbers.Number):
            self.size = (int(self.size), int(self.size))
        else:
            assert len(self.size) == 2, "Please provide only two dimensions (h, w) for size."
        image_width, image_height = clip[0].size
        crop_height, crop_width = self.size
        if crop_width > image_width or crop_height > image_height:
            msg = "Requested crop size {} is bigger than input size {}"
            raise ValueError(msg.format(self.size, (image_height, image_width)))

        return [img.crop((0, 0, crop_width, crop_height)) for img in clip]
        
        
class ClipThirdCrop(transforms.CenterCrop):
    """Crops the given list of PIL Image at the 3/3.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): Clip to be cropped.

        Returns:
            List of PIL Image: Cropped clip.
        """
        if isinstance(self.size, numbers.Number):
            self.size = (int(self.size), int(self.size))
        else:
            assert len(self.size) == 2, "Please provide only two dimensions (h, w) for size."
        image_width, image_height = clip[0].size
        crop_height, crop_width = self.size
        if crop_width > image_width or crop_height > image_height:
            msg = "Requested crop size {} is bigger than input size {}"
            raise ValueError(msg.format(self.size, (image_height, image_width)))

        return [img.crop((image_width - crop_width, image_height - crop_height, image_width, image_height)) for img in clip]


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ClipGaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, clip):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return [img.filter(ImageFilter.GaussianBlur(radius=sigma)) for img in clip]


class ClipDifference(object):
    def __call__(self, clip):
        length = len(clip)
        out_clip = []
        for idx in range(length):
            if idx == 0:
                out_img = clip[0] - clip[1]
            elif idx == length - 1:
                out_img = clip[length - 1] - clip[length - 2]
            else:
                out_img = clip[idx] - clip[idx - 1] / 2 - clip[idx + 1] / 2
            out_clip.append(out_img.detach())
        return out_clip


class ClipForeground(object):
    def __call__(self, clip):
        length = len(clip)
        out_clip = []
        for subclip_idx in range(length // 16):
            mean_img = 0
            for idx in range(subclip_idx * 16, (subclip_idx + 1) * 16):
                mean_img += clip[idx] / 16
            for idx in range(subclip_idx * 16, (subclip_idx + 1) * 16):
                out_img = clip[idx] - mean_img
                out_clip.append(out_img.detach())
        return out_clip


class ClipShuffle(object):
    def __call__(self, clip):
        length = len(clip)
        out_clip = []
        order = list(range(length // 4))
        random.shuffle(order)
        for idx in order:
            out_clip.append(clip[idx * 4 + 0])
            out_clip.append(clip[idx * 4 + 1])
            out_clip.append(clip[idx * 4 + 2])
            out_clip.append(clip[idx * 4 + 3])
        return out_clip


class ClipDifferenceToImage(object):
    def __call__(self, clip):
        length = len(clip)
        out_img = torch.zeros_like(clip[0])
        for idx in range(length // 2):
            out_img += clip[idx]
            out_img -= clip[length - 1 - idx]
        return [out_img.detach()]


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std, threed_data=False):
        self.threed_data = threed_data
        if self.threed_data:
            # convert to the proper format
            self.mean = torch.FloatTensor(mean).view(len(mean), 1, 1, 1)
            self.std = torch.FloatTensor(std).view(len(std), 1, 1, 1)
        else:
            self.mean = mean
            self.std = std

    def __call__(self, tensor):

        if self.threed_data:
            tensor.sub_(self.mean).div_(self.std)
        else:
            rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
            rep_std = self.std * (tensor.size()[0] // len(self.std))

            # TODO: make efficient
            for t, m, s in zip(tensor, rep_mean, rep_std):
                t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR

    Randomly select the smaller edge from the range of 'size'.
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        selected_size = np.random.randint(low=self.size[0], high=self.size[1] + 1, dtype=int)
        scale = GroupScale(selected_size, interpolation=self.interpolation)
        return scale(img_group)


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, num_crops=5, crop_idx=0, flip=False):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

        if num_crops not in [1, 3, 5, 10]:
            raise ValueError("num_crops should be in [1, 3, 5, 10] but ({})".format(num_crops))
        self.num_crops = num_crops
        self.crop_idx = crop_idx

        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        if self.num_crops == 3:
            w_step = (image_w - crop_w) // 4
            h_step = (image_h - crop_h) // 4
            offsets = list()
            if image_w != crop_w and image_h != crop_h:
                offsets.append((self.crop_idx * 2 * w_step, self.crop_idx * 2 * h_step))
                # offsets.append((0 * w_step, 0 * h_step))  # top
                # offsets.append((4 * w_step, 4 * h_step))  # bottom
                # offsets.append((2 * w_step, 2 * h_step))  # center
            else:
                if image_w < image_h:
                    offsets.append((2 * w_step, self.crop_idx * 2 * h_step))
                    # offsets.append((2 * w_step, 0 * h_step))  # top
                    # offsets.append((2 * w_step, 4 * h_step))  # bottom
                    # offsets.append((2 * w_step, 2 * h_step))  # center
                else:
                    offsets.append((self.crop_idx * 2 * w_step, 2 * h_step))
                    # offsets.append((0 * w_step, 2 * h_step))  # left
                    # offsets.append((4 * w_step, 2 * h_step))  # right
                    # offsets.append((2 * w_step, 2 * h_step))  # center

        else:
            # offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
            w_step = (image_w - crop_w) // 4
            h_step = (image_h - crop_h) // 4

            offsets = list()
            if self.crop_idx >= self.num_crops:
                raise NotImplementedError(f'crop_idx ({self.crop_idx}) do not support num_crops ({self.num_crops})')

            if self.crop_idx % 5 == 0:
                offsets.append((0, 0))  # upper left
            elif self.crop_idx % 5 == 1:
                offsets.append((4 * w_step, 0))  # upper right
            elif self.crop_idx % 5 == 2:
                offsets.append((0, 4 * h_step))  # lower left
            elif self.crop_idx % 5 == 3:
                offsets.append((4 * w_step, 4 * h_step))  # lower right
            elif self.crop_idx % 5 == 4:
                offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            if not self.flip:
                oversample_group.extend(normal_group)
            else:
                oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False, threed_data=False):
        self.roll = roll
        self.threed_data = threed_data

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.threed_data:
                return np.stack(img_group, axis=0)
            else:
                if self.roll:
                    return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
                else:
                    return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if len(pic.shape) == 4:
                # ((NF)xCxHxW) --> (Cx(NF)xHxW)
                img = torch.from_numpy(pic).permute(3, 0, 1, 2).contiguous()
            else:  # data is HW(FC)
                img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data
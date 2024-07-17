# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py,
published under an Apache License 2.0.

COMMENT FROM ORIGINAL:
Mixup and Cutmix
Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899) # NOQA
Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""

import numpy as np
import torch
#import cv2


def convert_to_one_hot(targets, num_classes, on_value=1.0, off_value=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        on_value (float): Target Value for ground truth class.
        off_value (float): Target Value for other classes.This value is used for
            label smoothing.
    """

    targets = targets.long().view(-1, 1)
    return torch.full(
        (targets.size()[0], num_classes), off_value, device=targets.device
    ).scatter_(1, targets, on_value)



def mixup_target(target, num_classes, lam=1.0, random_index=None, smoothing=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixup/cutmix.
        smoothing (float): Label smoothing value.
    """
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    target1 = convert_to_one_hot(
        target,
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )

    if isinstance(lam,torch.Tensor):

        target2 = convert_to_one_hot(
            target[random_index],
            num_classes,
            on_value=on_value,
            off_value=off_value,
        )
        lam = lam.cuda().unsqueeze(-1)
    else:
        target2 = convert_to_one_hot(
            target.flip(0),
            num_classes,
            on_value=on_value,
            off_value=off_value,
        )

    return target1 * lam + target2 * (1.0 - lam)




def rand_bbox(img_shape, lam, margin=0.0, count=None):
    """
    Generates a random square bbox based on lambda value.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def get_cutmix_bbox(img_shape, lam, correct_lam=True, count=None):
    """
    Generates the box coordinates for cutmix.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        correct_lam (bool): Apply lambda correction when cutmix bbox clipped by
            image borders.
        count (int): Number of bbox to generate
    """

    yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam



def saliency_bbox(vid, lam): #img size b c h w
    size = vid.size()
    b,c,t,h,w = size
    # vid = vid.transpose(1,2).reshape(b*t,c,h,w)
    #print(size)

    H = size[-2]
    W = size[-1]
    
    cut_rat = (1. - lam)**0.5
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    x_aixs = torch.linspace(0,2*(cut_w//2)-1,2*(cut_w//2))[None].expand(2*(cut_h//2),2*(cut_w//2)).cuda()
    y_aixs = torch.linspace(0,2*(cut_h//2)-1,2*(cut_h//2))[...,None].expand(2*(cut_h//2),2*(cut_w//2)).cuda()
    

    x_aixs = x_aixs[...,None]
    y_aixs = y_aixs[...,None]
    
    x_aixs = x_aixs[None].expand(b, 2*(cut_h//2), 2*(cut_w//2), 1)
    y_aixs = y_aixs[None].expand(b, 2*(cut_h//2), 2*(cut_w//2), 1)


    temp_imgs = vid[:,:,0,:,:].cpu().numpy().transpose(0, 2, 3, 1) #B H W C
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    saliencyMaps = [saliency.computeSaliency(img)[-1] for img in temp_imgs]
    saliencyMaps = torch.from_numpy(np.array(saliencyMaps))

    cor = torch.argmax(saliencyMaps.reshape(b,h*w),dim=1).cuda() #b
    y = cor//W
    x = cor%W

    y_aixs = torch.clip(y_aixs + y[:,None,None,None] - (cut_h//2), 0, H-1)
    x_aixs = torch.clip(x_aixs + x[:,None,None,None] - (cut_w//2), 0, W-1)

    crop_cor = torch.cat((y_aixs,x_aixs),dim=-1)

    bbx1 = torch.clip(x - cut_w // 2, 0, W)
    bby1 = torch.clip(y - cut_h // 2, 0, H)
    bbx2 = torch.clip(x + cut_w // 2, 0, W)
    bby2 = torch.clip(y + cut_h // 2, 0, H)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H*W))
    #print(bbx1.shape)

    return crop_cor.reshape(b, 2*(cut_h//2), 2*(cut_w//2), 2).long(), lam



class SaliencyCut:
    """
    Apply mixup and/or cutmix for videos at batch level.
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
        Features (https://arxiv.org/abs/1905.04899)
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        mix_prob=1.0,
        switch_prob=0.5,
        correct_lam=True,
        label_smoothing=0.1,
        num_classes=1000,
    ):
        """
        Args:
            mixup_alpha (float): Mixup alpha value.
            cutmix_alpha (float): Cutmix alpha value.
            mix_prob (float): Probability of applying mixup or cutmix.
            switch_prob (float): Probability of switching to cutmix instead of
                mixup when both are active.
            correct_lam (bool): Apply lambda correction when cutmix bbox
                clipped by image borders.
            label_smoothing (float): Apply label smoothing to the mixed target
                tensor. If label_smoothing is not used, set it to 0.
            num_classes (int): Number of classes for target.
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.correct_lam = correct_lam

    def _get_mixup_params(self):
        lam = 1.0
        use_cutmix = False
        if np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_batch(self, x):
        lam, use_cutmix = self._get_mixup_params()
        rand_index = None
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            if x.ndim == 4:
                b, c, h, w = x.shape
                x = x.reshape(b, c // 3, 3, h, w).permute(0, 2, 1, 3, 4)
            b, c, t, h, w = x.shape

            rand_index = torch.randperm(b).cuda()
            crop_cor, lam = saliency_bbox(x[rand_index], lam) # b h_box w_box 2
            _,h_box,w_box,_ = crop_cor.shape
            crop_cor_x = crop_cor[:,:,:,1].reshape(-1)
            crop_cor_y = crop_cor[:,:,:,0].reshape(-1)

            crop_cor_b = torch.ones(b,h_box*w_box).cuda() * torch.linspace(0,b-1,b)[:,None].cuda()
            crop_cor_b = crop_cor_b.reshape(-1).long()

            
            x[crop_cor_b,:,:, crop_cor_y, crop_cor_x] = x[rand_index][crop_cor_b,:,:, crop_cor_y, crop_cor_x]

            
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped)
        return lam, rand_index

    def __call__(self, x, target):
        assert len(x) > 1, "Batch size should be greater than 1 for mixup."
        lam, rand_index = self._mix_batch(x)
        target = mixup_target(
            target, self.num_classes, lam, rand_index, self.label_smoothing
        )
        return x, target
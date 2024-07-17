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
import torch.nn as nn
#import cv2
from PIL import Image, ImageDraw, ImageFont
from pytorchvideo.model import build_model
from pytorchvideo.utils import comm
from torch.nn.parallel import DistributedDataParallel
import copy
import torch.nn.functional as F
import os
from pytorchvideo.config import kfg

font = ImageFont.truetype('DejaVuSans.ttf', 15)

class MB_st(nn.Module):
    def __init__(self,in_channel,inter_channel) -> None:
        super(MB_st, self).__init__()
        self.query = nn.Linear(in_channel+1,inter_channel)
                               #,kernel_size=1,stride=1,padding=0)
        self.value = nn.Linear(
                in_channel+1,
                1,)
                # kernel_size=1,
                # stride=1)
        self.inter_channels = inter_channel


        #self.value.weight.data[0,:in_channel+1,0,0] = 0
        self.value.weight.data[0,in_channel] = 4
        self.value.bias.data[:] = 0

    def forward(self,x,lam,index,scale=32,use_cutmix=True):
        scale_s = scale[0] if isinstance(scale,(list,tuple)) else scale
        scale_t = scale[1] if isinstance(scale,(list,tuple)) else 1
        #print('scale_t',scale_t)
        if use_cutmix:
            if isinstance(x,list):
                x = x[0]
            n, t, c, h, w = x.size()
            x_lam  = x.reshape(n*t, c, h, w)
            x_lam_ = x.reshape(n, t, c, h, w)[index, :].reshape(n*t, c, h, w)
            lam_block = torch.zeros(n*t, 1, h, w).to(x_lam)
            lam_block[:] = lam-0.5
            x_lam  = torch.cat([x_lam, lam_block], dim=1)
            x_lam_ = torch.cat([x_lam_, -lam_block], dim=1)

            v_ = x_lam_
            v_ = self.value(v_.transpose(1,3)).transpose(1,3).contiguous().view(n*t, 1, -1)  # [N, 1, HxW]
            v_ = v_.permute(0, 2, 1)
            q_x = self.query(x_lam.transpose(1,3)).transpose(1,3).contiguous().view(  # q for lam: [N, HxW, C/r]
                n*t, self.inter_channels, -1).permute(0, 2, 1)
            k_x = self.query(x_lam_.transpose(1,3)).transpose(1,3).contiguous().view(n*t, self.inter_channels, -1)  # [N, C/r, HxW]
            pairwise_weight = torch.matmul(
                q_x, k_x
            )
            pairwise_weight /= q_x.shape[-1] ** 0.5
            pairwise_weight = pairwise_weight.softmax(dim=-1)
            mask_lam_ = torch.matmul(
                pairwise_weight.type(torch.float32), v_.type(torch.float32)
            ).view(n, t, 1, h, w)

            mask_lam_ = mask_lam_.expand(n, t, scale_t, h, w).reshape(n*t*scale_t, 1, h, w)
            
            mask_lam_ = F.interpolate(mask_lam_, scale_factor=scale_s, mode='nearest')

            mask_lam_ = torch.sigmoid(mask_lam_.type(torch.float32))
            mask = torch.cat([1 - mask_lam_, mask_lam_], dim=1) 
            #print(mask.shape)
            return mask,v_.reshape(n,t,1,h,w).expand(n,t,scale_t,h,w).reshape((n,t*scale_t,h,w))
        else:
            
            if isinstance(x,list):
                n, t, c, h, w = x[0].size()
                x = x[1]
                x_lam  = x # b c t
                x_lam_ = x[index] #
            else:
                
                n, t, c, h, w = x.size()
            
                x_lam  = x.mean(dim=[-1,-2]).permute(0,2,1) # b c t
                x_lam_ = x.mean(dim=[-1,-2]).permute(0,2,1)[index] #
            lam_block = torch.zeros(n, 1, t).to(x_lam)
            lam_block[:] = lam-0.5
            x_lam  = torch.cat([x_lam, lam_block], dim=1)
            x_lam_ = torch.cat([x_lam_, -lam_block], dim=1)

            v_ = x_lam_
            #print(v_.shape)
            v_ = self.value(v_.transpose(1,2)).transpose(1,2).contiguous().view(n, 1, -1)  # [N, 1, T]
            v_ = v_.permute(0, 2, 1)
            q_x = self.query(x_lam.transpose(1,2)).transpose(1,2).contiguous().view(  # q for lam: [N, T, C/r]
                n, self.inter_channels, -1).permute(0, 2, 1)
            k_x = self.query(x_lam_.transpose(1,2)).transpose(1,2).contiguous().view(n, self.inter_channels, -1)  # [N, C/r, T]
            pairwise_weight = torch.matmul(
                q_x, k_x
            )
            
            pairwise_weight /= q_x.shape[-1] ** 0.5
            pairwise_weight = pairwise_weight.softmax(dim=-1)
            mask_lam_ = torch.matmul(
                pairwise_weight.type(torch.float32), v_.type(torch.float32)
            ).view(n,t,1,1,1).expand(n,t,1,h,w)

            mask_lam_ = mask_lam_.expand(n, t, scale_t, h, w).reshape(n*t*scale_t, 1, h, w)

            mask_lam_ = F.interpolate(mask_lam_, scale_factor=scale_s, mode='nearest')

            mask_lam_ = torch.sigmoid(mask_lam_.type(torch.float32))
            mask = torch.cat([1 - mask_lam_, mask_lam_], dim=1)

            return mask,v_.reshape(n,t,1,1,1).expand(n,t,scale_t,1,1).reshape(n,t*scale_t,1,1)

class MB_t(nn.Module):
    def __init__(self,in_channel,inter_channel) -> None:
        super(MB_t, self).__init__()
        self.query = nn.Conv1d(in_channel+1,inter_channel,kernel_size=1,stride=1,padding=0)
        self.value = nn.Conv1d(
                in_channels=in_channel+1,
                out_channels=1,
                kernel_size=1,
                stride=1)
        self.inter_channels = inter_channel


        #self.value.weight.data[0,:in_channel+1,0,0] = 0
        self.value.weight.data[0,in_channel,0] = 4
        self.value.bias.data[:] = 0

    def forward(self,x,lam,index,scale=32):
        n, t, c, h, w = x.size()
     
        x_lam  = x.mean(dim=[-1,-2]).permute(0,2,1)
        x_lam_ = x.mean(dim=[-1,-2]).permute(0,2,1)[index] #
        lam_block = torch.zeros(n, 1, t).to(x_lam)
        lam_block[:] = lam-0.5
        x_lam  = torch.cat([x_lam, lam_block], dim=1)
        x_lam_ = torch.cat([x_lam_, -lam_block], dim=1)

        v_ = x_lam_
        #print(v_.shape)
        v_ = self.value(v_).view(n, 1, -1)  # [N, 1, T]
        v_ = v_.permute(0, 2, 1)
        q_x = self.query(x_lam).view(  # q for lam: [N, T, C/r]
            n, self.inter_channels, -1).permute(0, 2, 1)
        k_x = self.query(x_lam_).view(n, self.inter_channels, -1)  # [N, C/r, T]
        pairwise_weight = torch.matmul(
            q_x, k_x
        )
        
        pairwise_weight /= q_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        mask_lam_ = torch.matmul(
            pairwise_weight.type(torch.float32), v_.type(torch.float32)
        ).view(n, t,1,1).expand(n,t,h,w).reshape(n*t,1,h,w)
  
        mask_lam_ = F.interpolate(mask_lam_, scale_factor=scale, mode='nearest')

        mask_lam_ = torch.sigmoid(mask_lam_.type(torch.float32))
        mask = torch.cat([1 - mask_lam_, mask_lam_], dim=1)

        return mask,v_.reshape(n,t,1,1).expand(n,t,1,1)



# def mask_loss_func(mask, lam, num_seg):
#     """ loss for mixup masks """
    
#     assert mask.dim() == 4
#     n, k, h, w = mask.size()  # mixup mask [N, 2, H, W]
#     if k > 1:  # the second mask has no grad!
#         mask = mask[:, 1, :, :].unsqueeze(1) # bt 1 h w
#     m_mean = mask.reshape(n//num_seg,num_seg,h,w).sum(dim=[-1,-2]) / (h * w)  # mask mean in [0, 1], b t
#     m_mean = (m_mean - m_mean.mean(dim=[-1],keepdim=True)).abs() + m_mean.mean(dim=[-1],keepdim=True)
#     m_mean = m_mean.mean()

#     loss = torch.clamp(
#         torch.abs(1 - m_mean - lam) - 0.1, min=0.).mean()
    
#     if torch.isnan(loss):
#         print("Warming mask loss nan, mask sum: {}, skip.".format(mask))
#         # losses['loss'] = None
#         # self.overflow += 1
#         # if self.overflow > 10:
#         #     raise ValueError("Precision overflow in MixBlock, try fp32 training.")
#     return loss


def mask_loss_func(mask, lam):
    """ loss for mixup masks """
    
    assert mask.dim() == 4
    n, k, h, w = mask.size()  # mixup mask [N, 2, H, W]
    if k > 1:  # the second mask has no grad!
        mask = mask[:, 1, :, :].unsqueeze(1)
    m_mean = mask.sum() / (n * h * w)  # mask mean in [0, 1]

    loss = torch.clamp(
        torch.abs(1 - m_mean - lam) - 0.1, min=0.).mean()
    
    if torch.isnan(loss):
        print("Warming mask loss nan, mask sum: {}, skip.".format(mask))
        # losses['loss'] = None
        # self.overflow += 1
        # if self.overflow > 10:
        #     raise ValueError("Precision overflow in MixBlock, try fp32 training.")
    # print('temporal consistency off')
    return loss*0.1
    # return loss

def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    #print(comm.get_world_size())
    if comm.get_world_size() == 1:
    #    print('==1\n\n\n')
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, find_unused_parameters=False, **kwargs)
    #if fp16_compression:
    #    from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
    #    ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    #print('ddp\n\n\n')
    return ddp


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
    #print(random_index)
    #print(type(lam))
    if random_index != None:
        #print('lam tensor')
        
        target2 = convert_to_one_hot(
            target[random_index],
            num_classes,
            on_value=on_value,
            off_value=off_value,
        )
        if isinstance(lam,torch.Tensor):
            lam = lam.unsqueeze(-1)
    else:
        target2 = convert_to_one_hot(
            target.flip(0),
            num_classes,
            on_value=on_value,
            off_value=off_value,
        )
    
    #print(target1.shape, lam.shape)
    # print(lam)
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





class MaskMotionCut:
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
        cfg=None,
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
        # self.cutmix_alpha = cutmix_alpha
        self.cutmix_alpha = 1
        self.mix_prob = mix_prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.correct_lam = correct_lam
        # pretrained_tsm_dir = '/data1/tanyi/tsm_ucf101_exp0001_3b77f8e0.pth'
        #pretrained_tsm_dir = '/scratch/tanyi/best_45.7_cutmix_tsm_ssv1.pth'
        cfg_featuremap = copy.deepcopy(cfg)
        #cfg_featuremap._immutable(False)
        #cfg_featuremap.MODEL.NAME = 'TSM_ResNet_featuremap'

        model = build_model(cfg_featuremap)
        #ckpt = torch.load(pretrained_tsm_dir,map_location='cpu')['model']
        #model.load_state_dict(ckpt)
        
        model = create_ddp_model(model, broadcast_buffers=False)
        model = torch.compile(model)
        self.tsm_model = model
        #self.tsm_model.eval()
        self.tsm_model.requires_grad_(False)

        for n,m in self.tsm_model.named_modules():
            if 'drop' in n:
                m.eval()
                print(n)
        # try:
        #     self.tsm_model.module.drop.eval()
        # except AttributeError:
        #     self.tsm_model.drop.eval()

        self.softmax = torch.nn.Softmax(dim=-1)

        if cfg.MODEL.NAME in {'MVIT','VIDEO_SWIN'}:
            mix_box_s = MB_st(in_channel=768,inter_channel=768,)
        elif cfg.MODEL.NAME == 'Uniformer':
            mix_box_s = MB_st(in_channel=512,inter_channel=512,)
        elif cfg.MODEL.NAME == 'Vivit':
            mix_box_s = MB_st(in_channel=384,inter_channel=384,)
        else:
            mix_box_s = MB_st(in_channel=2048,inter_channel=2048//2,)
        
        self.mix_block_s = create_ddp_model(mix_box_s.cuda(), broadcast_buffers=False)

        # mix_box_t = MB_t(in_channel=2048,inter_channel=2048//2,)
        # self.mix_block_t = create_ddp_model(mix_box_t.cuda(), broadcast_buffers=False)

        self.momentum = 0.999
        #self.weight = 0
        #print(self.tsm_model.module.conv1.weight[0,0,0,:])
        #self.category = [i.replace('\n','') for i in open('/home/yit/ActionDataset/some_some_v1/category.txt')]
        #self.category = sorted(os.listdir('/home/yit/ActionDataset/ucf101/rgb'))
        self.iter_count = 0
        self.save_count = 0
        self.save_sample = 10
        dataset_size = 15000
        # dataset_size = 9537
        # dataset_size = 86017
        batch_size = 64
        iter_ep = dataset_size // batch_size
        self.save_point = iter_ep * 10
        self.output = cfg.OUTPUT_DIR

    def softmax_threshold(self,feature,T=0.3):
        bt,h,w = feature.shape
        
        feature = self.softmax(feature.reshape(bt,-1)/T)
        feature = feature > feature.mean(dim=[-1,-2])
        feature = feature * 1
        rand = torch.rand(feature.shape).to(feature) < 0.9  
        rand = rand * 1
        feature = feature * rand
        return feature
    
    
    def vis(self,vids,vids_ori,value,mask,target,index,lam,type,batch):
        root = 'vis_vid_'+type
        root = os.path.join(self.output,root)
        if not os.path.exists(root):
            os.mkdir(root)


        root = os.path.join(root,str(batch))
        if not os.path.exists(root):
            os.mkdir(root)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        b, c, h, w = vids.shape
        _,t,h_f,w_f = value.shape
        #print(vids.shape)
        vids = vids.reshape(b, c // 3, 3, h, w)
        vids_ori = vids_ori.reshape(b, c // 3, 3, h, w)

        vids = vids * torch.tensor(std).reshape(1,1,3,1,1).to(vids)  \
                    + torch.tensor(mean).reshape(1,1,3,1,1).to(vids) 
        vids_ori = vids_ori * torch.tensor(std).reshape(1,1,3,1,1).to(vids)  \
                    + torch.tensor(mean).reshape(1,1,3,1,1).to(vids) 
        
        
        vids = vids.permute(0,1,3,4,2).cpu() * 255
        vids_ori = vids_ori.permute(0,1,3,4,2).cpu() * 255 # b t h w c
        vids_ori_source = vids_ori[index.cpu()]

        black = torch.zeros((h,w)).to(vids).reshape(1,1,h,w,1).expand(b,c//3,h,w,3)
        mask_lam = mask[:,0].to(vids).reshape(b,c//3,1,h,w).expand(b,c//3,3,h,w).permute(0,1,3,4,2)*255
        mask_lam_ = mask[:,1].to(vids).reshape(b,c//3,1,h,w).expand(b,c//3,3,h,w).permute(0,1,3,4,2)*255

        vids = torch.cat((vids,black,black),dim=-2).detach().numpy()
        vids_ori = torch.cat((vids_ori,mask_lam,black),dim=-2).detach().numpy()


        v = value
        max = torch.max(value.reshape(b*t,h_f*w_f),dim=1)[0].reshape(b,t,1,1)
        min = torch.min(value.reshape(b*t,h_f*w_f),dim=1)[0].reshape(b,t,1,1)
        margin = max - min + 1e-6
        value = (value-min)/margin*254
        value = value.reshape(b,t,h_f,1,w_f,1,1).expand(b,t,h_f,h//h_f,w_f,w//w_f,3).reshape(b,t,h,w,3).to(black)

        vids_ori_source = torch.cat((vids_ori_source,mask_lam_,value),dim=-2).detach().numpy()

        current_lam = mask[:,0,:,:].mean(dim=[-1,-2]).reshape(b,c//3)


        vids = np.concatenate((vids_ori_source,vids_ori,vids),axis=2)

        for i,vid in enumerate(vids):
            
            dir = os.path.join(root,str(len(os.listdir(root))))
            if not os.path.exists(dir):
                os.mkdir(dir)
            for ii,img in enumerate(vid):
                
                # f = img.copy()
                # print(f.shape)
                img = Image.fromarray(img.astype(np.uint8))
                draw = ImageDraw.Draw(img)
                # draw.text((224,112), self.category[target[index][i]], font=font, fill=(0, 255, 0))
                # draw.text((224,336), self.category[target[i]], font=font, fill=(0, 255, 0))
            
                # for iii in range(h_f):
                #     for iiii in range(w_f):
                #         vv = v[i,ii,iii,iiii]
                #         if vv > 100:
                #             vv = str(int(vv))
                #         elif vv > 10:
                #             vv = "{:.1f}".format(vv)
                #         else:
                #             vv = "{:.2f}".format(vv)
                #         f = cv2.putText(f,vv,(448+iiii*32,iii*32+18),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
                vv = v[i].mean()
                draw.text((448+40,224+20), 'value mean {:.4f}'.format(vv), font=font, fill=(0, 255, 0))
                draw.text((224,560), 'setting lam: {}'.format(lam), font=font, fill=(0, 255, 0))
                draw.text((224,590), 'current lam: {}'.format(current_lam[i][ii]), font=font, fill=(0, 255, 0))
                draw.text((224,620), 'avg lam: {}'.format(current_lam[i].mean()), font=font, fill=(0, 255, 0))
                img.save(os.path.join(dir,'{}.jpg'.format(ii)))
    
           
        print('batch saved')



    @torch.no_grad()
    def momentum_update(self, model):
        """Momentum update of the k form q by hook, including the backbone and heads """
        # we don't update q to k when momentum > 1
        if self.momentum >= 1.:
            return
        # update k's backbone and cls head from q
        # print(self.tsm_model.device,'\n\n\n')
        for p_i, p_j in zip(self.tsm_model.parameters(),model.parameters()):
            p_i.data = p_i.data * self.momentum + \
                        p_j.data * (1. - self.momentum)


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




    def _mix_batch(self, x, target):
        lam, use_cutmix = self._get_mixup_params()
        rand_index_ol = None
        rand_index_ofl = None
        x_ofl = None
        lam_ofl = None
        mask_loss = None
        # if lam == 1.0:
        #     return 1.0
        # if use_cutmix:
            #print('maskmotion')
        #print(x.shape)

        x_clone = x.clone().detach()
        x_clone.requires_grad_(False)
        data={
            kfg.FRAMES: x_clone,
        }
        _,feature_map = self.tsm_model(data,featuremap='7x7')
        if isinstance(feature_map,list):
            feature_map[0]=feature_map[0].detach()
            feature_map[1]=feature_map[1].detach()
        else:
            feature_map = feature_map.detach()


        if x.ndim == 4:
            data_type = 'C'
            b, tc, h, w = x.shape
            c = 3
            t = tc//3
            x = x.reshape(b, t, c, h, w).reshape(-1,c,h,w) #.permute(0, 2, 1, 3, 4)
        elif x.ndim == 5:
            data_type = 'T'
            b, c, t, h, w = x.shape
            tc = t*c
            x = x.transpose(1,2).reshape(-1,c,h,w) #.permute(0, 2, 1, 3, 4)
        #print(x.shape)
        lam_ofl = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

        if isinstance(feature_map,list):
            _, c_f,t_f, h_f, w_f=feature_map[0].shape
            feature_map[0] = feature_map[0].transpose(1,2)
            
        else:
            if feature_map.ndim == 3:
                sp = feature_map.shape
                t_f = t // 2
                c_f = feature_map.shape[2]
                h_f = w_f = int((sp[1]//t_f)**0.5)
                feature_map = feature_map.reshape(b,t_f,h_f,w_f,c_f).permute(0,1,4,2,3)
                # print(feature_map.shape)
            elif feature_map.ndim == 5:
                # print(feature_map.shape)
                _, c_f,t_f, h_f, w_f=feature_map.shape
                feature_map = feature_map.transpose(1,2)
            elif feature_map.ndim == 4:
                bt, c_f, h_f, w_f=feature_map.shape
                t_f = t
                feature_map = feature_map.reshape(b, t, c_f, h_f, w_f)
        #feature_map = feature_map.mean(dim=1) # bt h w

        rand_index_ol = torch.randperm(b).cuda()
        rand_index_ofl = torch.randperm(b).cuda()

        #print(feature_map.shape)
       
        mask_ol_s = self.mix_block_s(feature_map,lam,rand_index_ol,(h//h_f,t//t_f),use_cutmix)[0].clone().detach()
        mask_ofl_s,value_s = self.mix_block_s(feature_map,lam_ofl,rand_index_ofl,(h//h_f,t//t_f),use_cutmix)
        
        # mask_ol_t = self.mix_block_t(feature_map,lam,rand_index_ol,h//h_f)[0].clone().detach()
        # mask_ofl_t,value_t = self.mix_block_t(feature_map,lam_ofl,rand_index_ofl,h//h_f)

        mask_ol = mask_ol_s
        mask_ofl = mask_ofl_s
        value = value_s

        mask_loss = mask_loss_func(mask_ofl,lam_ofl)

        
        if lam <=0.08 or (1-lam)<=0.08:
            mask_ol[:, 0, :, :] = lam
            mask_ol[:, 1, :, :] = 1 - lam

        #print(mask_ofl.shape,x.shape,feature_map.shape)
        x_ofl = x * mask_ofl[:, 0, :, :].unsqueeze(1) + x.reshape(b,-1,3,h,w)[rand_index_ofl, :].reshape(-1,3,h,w) * mask_ofl[:, 1, :, :].unsqueeze(1)
        x_ol = x * mask_ol[:, 0, :, :].unsqueeze(1) + x.reshape(b,-1,3,h,w)[rand_index_ol, :].reshape(-1,3,h,w) * mask_ol[:, 1, :, :].unsqueeze(1)

        x_ofl = x_ofl.reshape(b, tc, h, w)
        x_ol = x_ol.reshape(b, tc, h, w)

        if comm.get_rank() == 0:
            if self.iter_count % self.save_point == 0 and self.save_count < self.save_sample:
                #print(lam_ofl)
                if lam_ofl>0.3 and lam_ofl<0.7:
                #if True:
                    self.vis(x_ofl.detach(),x.detach(),value,mask_ofl,target,rand_index_ofl,lam_ofl,'ofl',self.iter_count)
                    #print(self.iter_count)
                self.save_count = self.save_count + 1
            else:
                if self.save_count == self.save_sample:
                    self.iter_count = self.iter_count + self.save_sample + 1
                    self.save_count = 0
                else:
                    self.iter_count = self.iter_count + 1

        x = x_ol

        #lam = mask_ol[:, 0].reshape(b,c//3,h,h).mean(dim=[-1,-2,-3])


        

        # saliency = int(use_cutmix)*self.softmax(feature_map.mean(2).reshape(b,c // 3,-1)).reshape(b,c // 3, h_f, w_f) + \
        #             (1-int(use_cutmix))*self.softmax(feature_map.mean(dim=[2,3,4])).reshape(b,c // 3, 1, 1).expand(b,c // 3, h_f, w_f)
        
        # all_sali = saliency.sum(dim=[-1,-2])
        # mask_low = mask_ol[:, 0, :, :].reshape(b,c // 3,h_f,h//h_f,w_f,w//w_f).mean(dim=[-1,-3]) # b t h_f, w_f
        # mask_low_ = 1 - mask_low
        # remain_sali = (saliency * mask_low).sum(dim=[-1,-2])/all_sali
        # source_sali = (saliency[rand_index_ol] * mask_low_).sum(dim=[-1,-2])/all_sali[rand_index_ol]
        # #source_sali = feature_map[rand_index][crop_cor_b,crop_cor_t,mask_y,mask_x].reshape(b,t,per_frame).sum(dim=[-1])/all_sali[rand_index]
        # lam_sali = remain_sali/(remain_sali+source_sali)
        # lam_sali = lam_sali.mean(dim=-1)

        if isinstance(feature_map,list):
            saliency = self.softmax(feature_map[0].mean(2).reshape(b,-1))\
            .reshape(b,t_f,1,h_f,w_f).expand(b,t_f,t//t_f,h_f,w_f).reshape(b,t,h_f,w_f)
        else:
            saliency = self.softmax(feature_map.mean(2).reshape(b,-1))\
                .reshape(b,t_f,1,h_f,w_f).expand(b,t_f,t//t_f,h_f,w_f).reshape(b,t,h_f,w_f)
        
        mask_low = mask_ol[:, 0, :, :].reshape(b,t,h_f,h//h_f,w_f,w//w_f).mean(dim=[-1,-3]) # b t h_f, w_f
        mask_low_ = 1 - mask_low
        remain_sali = (saliency * mask_low).sum(dim=[-1,-2,-3])
        source_sali = (saliency[rand_index_ol] * mask_low_).sum(dim=[-1,-2,-3])
        lam_sali = remain_sali/(remain_sali+source_sali)
        #lam_sali = lam_sali.mean(dim=-1)


        # print(lam)
        # print(lam_sali)
        # else:
        #     x_flipped = x.flip(0).mul_(1.0 - lam)
        #     x.mul_(lam).add_(x_flipped)
        if data_type=='T':
            x = x.reshape(b,t,c,h,w).transpose(1,2)
            x_ofl = x_ofl.reshape(b,t,c,h,w).transpose(1,2)

        return lam_sali, lam_ofl, rand_index_ol, rand_index_ofl, x, x_ofl, mask_loss

    def __call__(self, x, target):
        assert len(x) > 1, "Batch size should be greater than 1 for mixup."
        lam, lam_ofl, rand_index_ol, rand_index_ofl, x_ol, x_ofl, mask_loss = self._mix_batch(x,target)
        target_ol = mixup_target(
            target, self.num_classes, lam, rand_index_ol, self.label_smoothing
        )
        target_ofl = mixup_target(
            target, self.num_classes, lam_ofl, rand_index_ofl, self.label_smoothing
        )
        return x_ol, target_ol, x_ofl, target_ofl, mask_loss
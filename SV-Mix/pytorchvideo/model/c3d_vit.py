""" Change ViT to space-time domain
C3D-ViT: change patch-embedding to tubelet-embedding + spatio-temporal attention
by Zhaofan Qiu
"""
import logging
import torch
import torch.nn as nn
import numpy as np
import math
from collections import OrderedDict
from skimage import transform
import torch.nn.functional as F
from pytorchvideo.utils.weight_init import trunc_normal_
from .c2d_vit import Block, TubeletEmbed
from functools import partial
from .build import MODEL_REGISTRY  # register_model
from .base_net import BaseNet
from pytorchvideo.config import kfg, configurable


@MODEL_REGISTRY.register()
class C3D_ViT(BaseNet):
    _arch_dict = {
        'c3d_vit_t_p16': dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
        'c3d_vit_s_p16': dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
        'c3d_vit_b_p16': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
        'c3d_vit_b_p32': dict(patch_size=32, embed_dim=768, depth=12, num_heads=12),
        'c3d_vit_l_p16': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
        'c3d_vit_l_p32': dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16),
    }

    @configurable
    def __init__(self, img_size=224, early_stride=4, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 dropout_ratio=0., clip_length=16, weights='', transfer_weights=False, remove_fc=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.early_stride = early_stride
        self.patch_embed = TubeletEmbed(img_size=img_size, early_stride=early_stride, patch_size=patch_size,
                                        in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(self.dropout_ratio)
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.time_embed = nn.Parameter(torch.zeros(1, clip_length // early_stride, 1, embed_dim))
        nn.init.constant_(self.time_embed, 0)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # Load pretrained model
        if weights != '':
            self.load_pretrained(weights, transfer_weights, remove_fc, early_stride=early_stride,
                                 pos_embed_shape=self.pos_embed.shape, clip_length=clip_length)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        # Update ret dict with other configs
        # Will overwrite existing keys in ret
        ret.update({
            'dropout_ratio': cfg.MODEL.DROPOUT_RATIO,
            'early_stride': cfg.MODEL.EARLY_STRIDE,
            'img_size': cfg.TRANSFORM.CROP_SIZE,
            'clip_length': cfg.DATALOADER.NUM_SEGMENTS * cfg.DATALOADER.CLIP_LENGTH,
        })
        return ret

    @staticmethod
    def resize_pos_embed(posemb, pos_embed_shape, distilled):
        # Rescale the grid of position embeddings when loading from state_dict. Adapted from
        # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
        logger = logging.getLogger(__name__)
        logger.info('Resized position embedding: {:s} to {:s}'.format(str(posemb.shape), str(pos_embed_shape)))
        ntok_new = pos_embed_shape[1]
        if not distilled:
            posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
            ntok_new -= 1
        else:
            posemb_tok, posemb_grid = posemb[:, :2], posemb[0, 2:]
            ntok_new -= 2
        gs_old = int(math.sqrt(len(posemb_grid)))
        gs_new = int(math.sqrt(ntok_new))
        logger.info('Position embedding grid-size from {:s} to {:s}'.format(str(gs_old), str(gs_new)))
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb

    @staticmethod
    def checkpoint_filter_fn(state_dict, pos_embed_shape):
        """ convert patch embedding weight from manual patchify + linear proj to conv"""
        out_dict = {}
        if 'model' in state_dict:
            # For deit models
            state_dict = state_dict['model']
        for k, v in state_dict.items():
            if k == 'pos_embed' and v.shape != pos_embed_shape:
                # To resize pos embedding when using model at different size from pretrained weights
                v = C3D_ViT.resize_pos_embed(v, pos_embed_shape, False)
            out_dict[k] = v
        return out_dict

    @staticmethod
    def transfer_weights(state_dict, early_stride=None, *args, **kwargs):
        new_state_dict = {}
        for k, v in state_dict.items():
            v = v.detach().numpy()
            if k == 'patch_embed.proj.weight':
                shape = v.shape
                v = np.reshape(v, newshape=[shape[0], shape[1], 1, shape[2], shape[3]])
                if early_stride != 1:
                    s1 = early_stride // 2
                    s2 = early_stride - early_stride // 2 - 1
                    v = np.concatenate((np.zeros(shape=(shape[0], shape[1], s1, shape[2], shape[3])), v,
                                        np.zeros(shape=(shape[0], shape[1], s2, shape[2], shape[3]))), axis=2)
            elif k == 'time_embed':
                shape = v.shape  # [1, clip_length // early_stride, 1, embed_dim]
                v = np.reshape(v, newshape=[shape[1], shape[3]])
                tn = kwargs['clip_length'] // early_stride
                if tn != shape[1]:
                    v = transform.resize(v, (tn, shape[3]))
                v = np.reshape(v, newshape=[1, tn, 1, shape[3]])
            # change the name of head to fc
            elif 'head' in k:
                new_state_dict[k.replace('head', 'fc')] = torch.from_numpy(v)
            new_state_dict[k] = torch.from_numpy(v)

        return C3D_ViT.checkpoint_filter_fn(new_state_dict, kwargs['pos_embed_shape'])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, n_segment):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        nt, l, c = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, l, c)
        x = x + self.time_embed

        x = x.view(n_batch, n_segment * l, c)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = x[kfg.FRAMES]
        bsz = x.size(0)
        chn = x.size(1)
        lgt = x.size(2)
        hig = x.size(3)
        wid = x.size(4)
        x = x.view(bsz, chn, lgt // self.early_stride, self.early_stride, hig, wid).transpose(1, 2)
        x = x.reshape(bsz * lgt // self.early_stride, chn, self.early_stride, hig, wid)

        x = self.forward_features(x, lgt // self.early_stride)
        x = self.drop(x)
        x = self.fc(x)
        y = x
        if self.training:
            return y
        else:
            return y.softmax(dim=1)
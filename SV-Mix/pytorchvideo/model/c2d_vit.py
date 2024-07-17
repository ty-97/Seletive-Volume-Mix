"""
C2D-VIT: only change patch-embedding to tubelet-embedding
This implementation is based on https://github.com/rwightman/pytorch-image-models
Modified by Zhaofan Qiu
zhaofanqiu@gmail.com
"""
import logging
import torch
import torch.nn as nn
import numpy as np
import math
from collections import OrderedDict
import torch.nn.functional as F
from pytorchvideo.utils.vit_helpers import to_2tuple
from pytorchvideo.utils.weight_init import trunc_normal_
from pytorchvideo.layers.drop import DropPath
from functools import partial
from .build import MODEL_REGISTRY  # register_model
from .base_net import BaseNet
from pytorchvideo.config import kfg, configurable


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TubeletEmbed(nn.Module):
    """ Video to Tubelet Embedding
    """
    def __init__(self, img_size=224, early_stride=4, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # fixed with time-length=time-stride=4
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.early_stride = early_stride

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(early_stride,) + patch_size, stride=(early_stride,) + patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert T == self.early_stride and H == self.img_size[0] and W == self.img_size[1],  \
            f"Input image size ({T}*{H}*{W}) doesn't match model ({self.early_stride}*{self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


@MODEL_REGISTRY.register()
class C2D_ViT(BaseNet):
    """ Vision Transformer

        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
            https://arxiv.org/abs/2010.11929
        """
    _arch_dict = {
        'c2d_vit_t_p16': dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
        'c2d_vit_s_p16': dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
        'c2d_vit_b_p16': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
        'c2d_vit_b_p32': dict(patch_size=32, embed_dim=768, depth=12, num_heads=12),
        'c2d_vit_l_p16': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
        'c2d_vit_l_p32': dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16),
    }

    @configurable
    def __init__(self, img_size=224, early_stride=4, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 dropout_ratio=0., weights='', transfer_weights=False, remove_fc=False):
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

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # Load pretrained model
        if weights != '':
            self.load_pretrained(weights, transfer_weights, remove_fc, early_stride=early_stride,
                                 pos_embed_shape=self.pos_embed.shape)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        # Update ret dict with other configs
        # Will overwrite existing keys in ret
        ret.update({
            'dropout_ratio': cfg.MODEL.DROPOUT_RATIO,
            'early_stride': cfg.MODEL.EARLY_STRIDE,
            'img_size': cfg.TRANSFORM.CROP_SIZE
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
                v = C2D_ViT.resize_pos_embed(v, pos_embed_shape, False)
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
            # change the name of head to fc
            elif 'head' in k:
                new_state_dict[k.replace('head', 'fc')] = torch.from_numpy(v)
            new_state_dict[k] = torch.from_numpy(v)

        return C2D_ViT.checkpoint_filter_fn(new_state_dict, kwargs['pos_embed_shape'])

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

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

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

        x = self.forward_features(x)
        x = self.drop(x)
        x = self.fc(x)
        y = x.view(bsz, lgt // self.early_stride, -1).mean(dim=1)
        if self.training:
            return y
        else:
            return y.softmax(dim=1)

import math
import logging
import weakref

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from einops import rearrange

from .build import MODEL_REGISTRY
from .base_net import BaseNet
from pytorchvideo.config import kfg, configurable
from pytorchvideo.layers.drop import DropPath
from pytorchvideo.utils.vit_helpers import to_2tuple

try:
    from torch.nn.init import trunc_normal_
except ImportError as e:
    from pytorchvideo.utils.weight_init import trunc_normal_

__all__ = ['SwinTransformer']


def window_partition(x, window_size):
    """
    Part input shape to window_size
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse input shape from window_size to original size
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """
    MLP module
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super(Mlp, self).__init__()
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


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
        It supports both of shifted and non-shifted window.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        Returns:
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple), B_, nH, N, C // nH

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, f"shift_size must in 0-window_size," \
                                                        f" got shift_size: {self.shift_size}," \
                                                        f" window_size: {self.window_size}"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, N, N
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward_attn(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        return x

    def forward_mlp(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_attn, x)
        else:
            x = self.forward_attn(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_mlp, x)
        else:
            x = x + self.forward_mlp(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_checkpoint=use_checkpoint)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size tuple(int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


@MODEL_REGISTRY.register()
class SwinTransformer(BaseNet):
    r""" Swin Transformer from
    `"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" <https://arxiv.org/pdf/2103.14030.pdf>`_

    Args:
        duration (int): Num of clips
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        last_drop_rate (float): Dropout rate before fc layer. Default: 0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        super_img_rows (int): Spatial layout of input clips
        early_stride (int): Clip length
        image_mode (str): For preprocess of input features
        spec_alpha (float): Argument for motion enhancement operation
        weights (str): Path to pretrained weight file
        transfer_weights (bool): Whether transfer weights of pretrained model
        remove_fc (bool): Whether remove the last fc layer of pretrained model
    """
    _arch_dict = {
        # MSImageFormer-B1
        # ImageNet-22k weights: ``swin_base_patch4_window7_224_22k`` from
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
        'msif_base_192_3x3': dict(img_size=192, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                  window_size=[12, 12, 12, 18], super_img_rows=3),
        # MSImageFormer-B2
        # ImageNet-22k weights ``swin_base_patch4_window7_224_22k`` from
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
        'msif_base_192_4x4': dict(img_size=192, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                  window_size=[12, 12, 12, 24], super_img_rows=4),
        # MSImageFormer-B3
        # ImageNet-22k weights ``swin_base_patch4_window12_384_22k`` from
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
        'msif_base_384_3x3': dict(img_size=384, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                  window_size=[12, 12, 12, 36], super_img_rows=3),
        # MSImageFormer-L1
        # ImageNet-22k weights ``swin_large_patch4_window7_224_22k`` from
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
        'msif_large_192_3x3': dict(img_size=192, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                                   window_size=[12, 12, 12, 18], super_img_rows=3),
        # MSImageFormer-L2
        # ImageNet-22k weights ``swin_large_patch4_window7_224_22k`` from
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
        'msif_large_192_4x4': dict(img_size=192, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                                   window_size=[12, 12, 12, 24], super_img_rows=4),
        # MSImageFormer-L3
        # ImageNet-22k weights ``swin_large_patch4_window12_384_22k`` from
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
        'msif_large_384_3x3': dict(img_size=192, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                                   window_size=[12, 12, 12, 36], super_img_rows=3),
    }

    @configurable
    def __init__(self, duration=8, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, last_drop_rate=0.,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, super_img_rows=1, early_stride=1,
                 image_mode=None, spec_alpha=1., weights='', transfer_weights=False, remove_fc=False):
        super().__init__()
        self.duration = duration
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.super_img_rows = super_img_rows
        self.early_stride = early_stride

        self.img_size = img_size
        self.patch_size = patch_size
        self.window_size = [window_size for _ in depths] if not isinstance(window_size, list) else window_size
        self.image_mode = image_mode
        self.spec_alpha = spec_alpha

        self.frame_padding = self.duration % super_img_rows if self.image_mode else 0
        if self.frame_padding != 0:
            self.frame_padding = self.super_img_rows - self.frame_padding
            self.duration += self.frame_padding

        # split image into non-overlapping patches
        if self.image_mode:
            super_img_dim = (super_img_rows, self.duration // super_img_rows)
            super_img_size = (img_size * super_img_dim[0], img_size * super_img_dim[1])
        else:
            super_img_size = (img_size, img_size)

        self.patch_embed = PatchEmbed(
            img_size=super_img_size, patch_size=patch_size, in_chans=in_chans*self.early_stride,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=self.window_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=last_drop_rate) if last_drop_rate > 0 else nn.Identity()
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # Load pretrained model
        if weights != '':
            self.load_pretrained(weights, transfer_weights, remove_fc, model=weakref.proxy(self))

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        # Update ret dict with other configs
        # Will overwrite existing keys in ret
        arch_cfgs = cfg.MODEL.ARCH_CFGS
        ret.update({
            'duration': int(arch_cfgs[1]),
            'drop_path_rate': cfg.MODEL.DROP_PATH_RATIO,
            'last_drop_rate': cfg.MODEL.DROPOUT_RATIO,
            'early_stride': cfg.MODEL.EARLY_STRIDE,
            'image_mode': arch_cfgs[2],
            'spec_alpha': float(arch_cfgs[3]),
            'use_checkpoint': True if arch_cfgs[4] == 'checkpoint' else False,
        })
        return ret

    @staticmethod
    def transfer_weights(state_dict, model=None, *args, **kwargs):
        new_state_dict = state_dict.copy()
        '''
        ## Resizing the positional embeddings in case they don't match
        if img_size != cfg['input_size'][1]:
            pos_embed = state_dict['pos_embed']
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode='nearest')
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            state_dict['pos_embed'] = new_pos_embed
        '''
        for key in state_dict:
            if 'patch_embed.proj.weight' in key:
                pretrained_weight = state_dict[key]
                shape = pretrained_weight.shape
                model_in_chans = model.patch_embed.in_chans
                if shape[1] != model_in_chans:
                    early_stride = model_in_chans // shape[1]
                    s1 = early_stride // 2
                    s2 = early_stride - s1 - 1
                    v = torch.cat((torch.zeros(shape[0], s1 * shape[1], shape[2], shape[3]),
                                   pretrained_weight,
                                   torch.zeros(shape[0], s2 * shape[1], shape[2], shape[3])), dim=1)
                    new_state_dict[key] = v

            if 'attn_mask' in key:
                del new_state_dict[key]

            # if window_size != pretrained_window_size:
            if 'relative_position_index' in key:
                del new_state_dict[key]

            # resize relative position bias table
            if 'relative_position_bias_table' in key:
                pretrained_table = state_dict[key]
                pretrained_table_size = int(math.sqrt(pretrained_table.shape[0]))
                table_size = int(math.sqrt(model.state_dict()[key].shape[0]))
                if pretrained_table_size != table_size:
                    table = pretrained_table.permute(1, 0).view(1, -1, pretrained_table_size, pretrained_table_size)
                    table = nn.functional.interpolate(table, size=table_size, mode='bilinear')
                    table = table.view(-1, table_size * table_size).permute(1, 0)
                    new_state_dict[key] = table

            # change the name of head to fc
            if 'head' in key:
                new_state_dict[key.replace('head', 'fc')] = new_state_dict.pop(key)

        return new_state_dict

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_super_img(self, x):
        input_size = x.shape[-2:]
        if input_size != to_2tuple(self.img_size):
            x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')
        x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=self.super_img_rows, c=3*self.early_stride)
        return x

    def create_specauga_img(self, x, alpha=1.):
        input_size = x.shape[-2:]
        if input_size != to_2tuple(self.img_size):
            x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')
        B, C_, H, W = x.shape
        x = x.view(B, self.duration, self.early_stride, 3, H, W)
        spec = torch.fft.rfft(x, dim=2, norm='ortho')
        len_spec = spec.shape[2]
        alpha = torch.tensor([alpha])
        w = torch.stack([1 - alpha + (2 * alpha * k)/(len_spec - 1) for k in range(len_spec)], dim=-1)
        w = w.view(w.shape[0], 1, w.shape[1], 1, 1, 1).to(x.device)
        spec = spec * w
        x = torch.fft.irfft(spec, n=self.early_stride, dim=2, norm='ortho')
        x = rearrange(x, 'b (th tw) c_t c h w -> b (c_t c) (th h) (tw w)', th=self.super_img_rows)
        return x

    def create_trunc_img(self, x, alpha):
        input_size = x.shape[-2:]
        if input_size != to_2tuple(self.img_size):
            x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')
        B, C_, H, W = x.shape
        x = x.view(B, self.duration, self.early_stride, 3, H, W)
        alpha = int(alpha) + 1
        spec = torch.fft.rfft(x, dim=2, norm='ortho')
        spec[:, :, :alpha] = spec[:, :, :alpha] * 0
        x = torch.fft.irfft(spec, n=self.early_stride, dim=2, norm='ortho')
        x = rearrange(x, 'b (th tw) c_t c h w -> b (c_t c) (th h) (tw w)', th=self.super_img_rows)
        return x

    def pad_frames(self, x):
        frame_num = self.duration - self.frame_padding
        x = x.view((-1, 3 * self.early_stride * frame_num) + x.size()[2:])  # BxN, TxC, H, W
        x_padding = torch.zeros((x.shape[0], 3 * self.early_stride * self.frame_padding) + x.size()[2:]).to(x.device)
        x = torch.cat((x, x_padding), dim=1)
        assert x.shape[1] == 3 * self.early_stride * self.duration, \
            'frame number %d not the same as adjusted input size %d' % (x.shape[1], 3 * self.duration)

        return x

    def forward_features(self, x, ret_map=False):
        #        x = rearrange(x, 'b (t c) h w -> b c h (t w)', t=self.duration)
        # in evaluation, it's Bx(num_crops*num_cips*num_frames*3)xHxW

        if self.frame_padding > 0:
            x = self.pad_frames(x)
        else:
            x = x.view((-1, 3 * self.early_stride * self.duration) + x.size()[2:])

        if self.image_mode == 'super':
            x = self.create_super_img(x)
        elif self.image_mode == 'spau':
            x = self.create_specauga_img(x, self.spec_alpha)
        elif self.image_mode == 'trunc':
            x = self.create_trunc_img(x, alpha=self.spec_alpha)
        else:
            x = rearrange(x, 'b (n t c) h w -> (b n t) c h w', t=self.duration, c=3)

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x_map = x
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        if ret_map:
            return x_map, x
        else:
            return x

    def forward(self, x, ret_map=False):
        ret_x = self.forward_features(x[kfg.FRAMES], ret_map)
        x_map = None
        if ret_map:
            x_map, x = ret_x
        else:
            x = ret_x
        x = self.dropout(x)
        x = self.fc(x)
        if not self.image_mode:
            x = x.view(-1, self.duration, self.num_classes)
            x = torch.mean(x, dim=1)
        if ret_map:
            return [x_map, x]
        elif self.training:
            return x
        else:
            return x.softmax(dim=1)

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

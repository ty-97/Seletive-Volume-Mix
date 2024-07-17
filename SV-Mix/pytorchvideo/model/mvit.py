import math
import weakref
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from functools import partial
from .build import MODEL_REGISTRY
from .base_net import BaseNet
from pytorchvideo.config import kfg, configurable
from pytorchvideo.layers.drop import DropPath
from pytorchvideo.utils.weight_init import trunc_normal_

logger = logging.getLogger(__name__)

__all__ = ['MVIT']


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    if verbose:
        logger.info(f"min width {min_width}")
        logger.info(f"width {width} divisor {divisor}")
        logger.info(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size**2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(1, 16, 16),
        stride=(1, 4, 4),
        padding=(1, 7, 7),
        conv_2d=False,
    ):
        super().__init__()
        if conv_2d:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.proj = conv(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, keep_spatial=False):
        x = self.proj(x)
        if keep_spatial:
            return x, x.shape
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2), x.shape


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    )

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


def get_rel_pos(rel_pos, d):
    if isinstance(d, int):
        ori_d = rel_pos.shape[0]
        if ori_d == d:
            return rel_pos
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate(
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode="linear",
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)


def cal_rel_pos_spatial(
    attn, q, k, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w
):
    """
    Decomposed Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio
        - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio
        - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    # Intepolate rel pos if needed.
    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw)
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    rel_h_q = torch.einsum(
        "bythwc,hkc->bythwk", r_q, Rh
    )  # [B, H, q_t, qh, qw, k_h]
    rel_w_q = torch.einsum(
        "bythwc,wkc->bythwk", r_q, Rw
    )  # [B, H, q_t, qh, qw, k_w]

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel_h_q[:, :, :, :, :, None, :, None]
        + rel_w_q[:, :, :, :, :, None, None, :]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


def cal_rel_pos_temporal(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t):
    """
    Temporal Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dt = int(2 * max(q_t, k_t) - 1)
    # Intepolate rel pos if needed.
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    # Scale up rel pos if shapes for q and k are different.
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = (
        torch.arange(q_t)[:, None] * q_t_ratio
        - torch.arange(k_t)[None, :] * k_t_ratio
    )
    dist_t += (k_t - 1) * k_t_ratio
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(
        q_t, B * n_head * q_h * q_w, dim
    )

    # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
    rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
    rel = rel.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel[:, :, :, :, :, :, None, None]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        # If True, perform pool before projection.
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        separate_qkv=False,
    ):
        super().__init__()
        self.pool_first = pool_first
        self.separate_qkv = separate_qkv
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        self.mode = mode
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if pool_first or separate_qkv:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q = ()
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            self.pool_q = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        if self.rel_pos_spatial:
            assert input_size[1] == input_size[2]
            size = input_size[1]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)
        if self.rel_pos_temporal:
            self.rel_pos_t = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim)
            )
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_t, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, thw_shape):
        B, N, _ = x.shape

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"
            if not self.separate_qkv:
                qkv = (
                    self.qkv(x)
                    .reshape(B, N, 3, self.num_heads, -1)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
            else:
                q = k = v = x
                q = (
                    self.q(q)
                    .reshape(B, N, self.num_heads, -1)
                    .permute(0, 2, 1, 3)
                )
                k = (
                    self.k(k)
                    .reshape(B, N, self.num_heads, -1)
                    .permute(0, 2, 1, 3)
                )
                v = (
                    self.v(v)
                    .reshape(B, N, self.num_heads, -1)
                    .permute(0, 2, 1, 3)
                )

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        if self.pool_first:
            q_N = (
                np.prod(q_shape) + 1
                if self.has_cls_embed
                else np.prod(q_shape)
            )
            k_N = (
                np.prod(k_shape) + 1
                if self.has_cls_embed
                else np.prod(k_shape)
            )
            v_N = (
                np.prod(v_shape) + 1
                if self.has_cls_embed
                else np.prod(v_shape)
            )

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = (
                self.q(q)
                .reshape(B, q_N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = (
                self.v(v)
                .reshape(B, v_N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = (
                self.k(k)
                .reshape(B, k_N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

        N = q.shape[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                k,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
            )

        if self.rel_pos_temporal:
            attn = cal_rel_pos_temporal(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_t,
            )
        attn = attn.softmax(dim=-1)

        x = attn @ v

        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        input_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        layer_scale_init_value=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        dim_mul_in_att=False,
        separate_qkv=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        att_dim = dim_out if dim_mul_in_att else dim
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_temporal=rel_pos_temporal,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
            separate_qkv=separate_qkv,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim_out)),
                requires_grad=True,
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(stride_skip) > 0 and np.prod(stride_skip) > 1
            else None
        )

    def forward(self, x, thw_shape=None):
        x_norm = self.norm1(x)
        x_block, thw_shape_new = self.attn(x_norm, thw_shape)
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        if self.gamma_1 is not None:
            x = x_res + self.drop_path(self.gamma_1 * x_block)
        else:
            x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * x_mlp)
        else:
            x = x + self.drop_path(x_mlp)
        if thw_shape:
            return x, thw_shape_new
        else:
            return x


@MODEL_REGISTRY.register()
class MVIT(BaseNet):
    """MViT & MViTv2
    PyTorch implementation of
    - MViT: `Multiscale Vision Transformers` - https://arxiv.org/abs/2104.11227
    - MViTv2: `MViTv2: Improved Multiscale Vision Transformers for Classification and Detection` -
     https://arxiv.org/abs/2112.01526)

    """
    _arch_dict = {
        # MViTv2 model config for video action recognition
        'mvitv2_t_3d': dict(patch_2d=False, embed_dim=96, num_heads=1, depth=10, cls_embed_on=False, use_abs_pos=False,
                            rel_pos_spatial=True, rel_pos_temporal=True, residual_pooling=True, dim_mul_in_att=True,
                            zero_decay_pos_cls=False,
                            patch_kernel=[3, 7, 7], patch_stride=[2, 4, 4], patch_padding=[1, 3, 3],
                            dim_mul_=[[1, 2.0], [3, 2.0], [8, 2.0]],
                            head_mul_=[[1, 2.0], [3, 2.0], [8, 2.0]],
                            pool_kvq_kernel=[3, 3, 3], pool_kv_stride_adaptive=[1, 8, 8],
                            pool_q_stride=[[0, 1, 1, 1],

                                           [1, 1, 2, 2], [2, 1, 1, 1],

                                           [3, 1, 2, 2], [4, 1, 1, 1], [5, 1, 1, 1],
                                           [6, 1, 1, 1], [7, 1, 1, 1],

                                           [8, 2, 2, 2], [9, 1, 1, 1]]),
        'mvitv2_s_3d': dict(patch_2d=False, embed_dim=96, num_heads=1, depth=16, cls_embed_on=False, use_abs_pos=False,
                            rel_pos_spatial=True, rel_pos_temporal=True, residual_pooling=True, dim_mul_in_att=True,
                            zero_decay_pos_cls=False,
                            patch_kernel=[3, 7, 7], patch_stride=[2, 4, 4], patch_padding=[1, 3, 3],
                            dim_mul_=[[1, 2.0], [3, 2.0], [14, 2.0]],
                            head_mul_=[[1, 2.0], [3, 2.0], [14, 2.0]],
                            pool_kvq_kernel=[3, 3, 3], pool_kv_stride_adaptive=[1, 8, 8],
                            pool_q_stride=[[0, 1, 1, 1],

                                           [1, 1, 2, 2], [2, 1, 1, 1],

                                           [3, 1, 2, 2], [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1],
                                           [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1],
                                           [12, 1, 1, 1], [13, 1, 1, 1],

                                           [14, 1, 2, 2], [15, 1, 1, 1]]),
        'mvitv2_b_3d': dict(patch_2d=False, embed_dim=96, num_heads=1, depth=24, cls_embed_on=False, use_abs_pos=False,
                            rel_pos_spatial=True, rel_pos_temporal=True, residual_pooling=True, dim_mul_in_att=True,
                            zero_decay_pos_cls=False,
                            patch_kernel=[3, 7, 7], patch_stride=[2, 4, 4], patch_padding=[1, 3, 3],
                            dim_mul_=[[2, 2.0], [5, 2.0], [21, 2.0]],
                            head_mul_=[[2, 2.0], [5, 2.0], [21, 2.0]],
                            pool_kvq_kernel=[3, 3, 3], pool_kv_stride_adaptive=[1, 8, 8],
                            pool_q_stride=[[0, 1, 1, 1], [1, 1, 1, 1],

                                           [2, 1, 2, 2], [3, 1, 1, 1], [4, 1, 1, 1],

                                           [5, 1, 2, 2], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1, 1],
                                           [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1],
                                           [13, 1, 1, 1], [14, 1, 1, 1], [15, 1, 1, 1], [16, 1, 1, 1],
                                           [17, 1, 1, 1], [18, 1, 1, 1], [19, 1, 1, 1], [20, 1, 1, 1],

                                           [21, 1, 2, 2], [22, 1, 1, 1], [23, 1, 1, 1]]),
        'mvitv2_l_3d': dict(patch_2d=False, embed_dim=144, num_heads=2, depth=48, cls_embed_on=False, use_abs_pos=False,
                            rel_pos_spatial=True, rel_pos_temporal=True, residual_pooling=True, dim_mul_in_att=True,
                            zero_decay_pos_cls=False,
                            patch_kernel=[3, 7, 7], patch_stride=[2, 4, 4], patch_padding=[1, 3, 3],
                            dim_mul_=[[2, 2.0], [8, 2.0], [44, 2.0]],
                            head_mul_=[[2, 2.0], [8, 2.0], [44, 2.0]],
                            pool_kvq_kernel=[3, 3, 3], pool_kv_stride_adaptive=[1, 8, 8],
                            pool_q_stride=[[0, 1, 1, 1], [1, 1, 1, 1],

                                           [2, 1, 2, 2], [3, 1, 1, 1], [4, 1, 1, 1],
                                           [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1],

                                           [8, 1, 2, 2], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1],
                                           [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 1, 1], [15, 1, 1, 1],
                                           [16, 1, 1, 1], [17, 1, 1, 1], [18, 1, 1, 1], [19, 1, 1, 1],
                                           [20, 1, 1, 1], [21, 1, 1, 1], [22, 1, 1, 1], [23, 1, 1, 1],
                                           [24, 1, 1, 1], [25, 1, 1, 1], [26, 1, 1, 1], [27, 1, 1, 1],
                                           [28, 1, 1, 1], [29, 1, 1, 1], [30, 1, 1, 1], [31, 1, 1, 1],
                                           [32, 1, 1, 1], [33, 1, 1, 1], [34, 1, 1, 1], [35, 1, 1, 1],
                                           [36, 1, 1, 1], [37, 1, 1, 1], [38, 1, 1, 1], [39, 1, 1, 1],
                                           [40, 1, 1, 1], [41, 1, 1, 1], [42, 1, 1, 1], [43, 1, 1, 1],

                                           [44, 1, 2, 2], [45, 1, 1, 1], [46, 1, 1, 1], [47, 1, 1, 1]])
    }

    @configurable
    def __init__(self, patch_2d, patch_stride, temporal_size, spatial_size, in_chans, num_classes, embed_dim, num_heads,
                 depth, drop_path_rate, cls_embed_on, use_abs_pos, rel_pos_spatial, rel_pos_temporal, patch_kernel,
                 patch_padding, mlp_ratio=4., qkv_bias=True, drop_rate=0., dropout_rate=0., head_init_scale=1.,
                 mode='conv', norm='layernorm', use_mean_pooling=False, use_fixed_sincos_pos=False, sep_pos_embed=False,
                 dim_mul_=None, head_mul_=None, pool_q_stride=None, pool_kv_stride=None, pool_kvq_kernel=None,
                 pool_kv_stride_adaptive=None, norm_stem=False, dim_mul_in_att=False, rel_pos_zero_init=False,
                 residual_pooling=False, separate_qkv=False, act_checkpoint=False, zero_decay_pos_cls=False,
                 weights='', transfer_weights=False, remove_fc=False):
        super().__init__()
        if dim_mul_ is None:
            dim_mul_ = []
        if head_mul_ is None:
            head_mul_ = []
        if pool_q_stride is None:
            pool_q_stride = []
        if pool_kv_stride is None:
            pool_kv_stride = []
        if pool_kv_stride_adaptive is None:
            pool_kv_stride_adaptive = []

        # Prepare input.
        self.use_2d_patch = patch_2d
        self.patch_stride = patch_stride
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = temporal_size // self.patch_stride[0]
        self.H = spatial_size // self.patch_stride[1]
        self.W = spatial_size // patch_stride[2]
        # Prepare output.
        self.drop_rate = drop_rate
        self.cls_embed_on = cls_embed_on
        self.use_mean_pooling = use_mean_pooling
        # Params for positional embedding
        self.use_abs_pos = use_abs_pos
        self.use_fixed_sincos_pos = use_fixed_sincos_pos
        self.sep_pos_embed = sep_pos_embed
        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        self.zero_decay_pos_cls = zero_decay_pos_cls
        self.act_checkpoint = act_checkpoint
        if norm == 'layernorm':
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm")
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            conv_2d=self.use_2d_patch
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]  # Shape of input image/video
        assert  self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]  # Shape of patch embedded feature
        num_patches = math.prod(self.patch_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim)
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, pos_embed_dim, embed_dim),
                    requires_grad=not self.use_fixed_sincos_pos,
                )
        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(dim_mul_)):
            dim_mul[dim_mul_[i][0]] = dim_mul_[i][1]
        for i in range(len(head_mul_)):
            head_mul[head_mul_[i][0]] = head_mul_[i][1]

        pool_q = [[] for i in range(depth)]
        pool_kv = [[] for i in range(depth)]
        stride_q = [[] for i in range(depth)]
        stride_kv = [[] for i in range(depth)]

        # Get kernel and stride of Pool_Q in each block
        for i in range(len(pool_q_stride)):
            stride_q[pool_q_stride[i][0]] = pool_q_stride[i][1:]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_q_stride[i][i:]
                ]

        # Get stride of Pool_K and Pool_V in each block
        # If pool_kv_stride_adaptive is not None, initialize pool_kv_stride
        if pool_kv_stride_adaptive is not None:
            _stride_kv = pool_kv_stride_adaptive
            pool_kv_stride = []
            for i in range(depth):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_stride.append([i] + _stride_kv)

        # Get kernel of Pool_K and Pool_V in each block
        for i in range(len(pool_kv_stride)):
            stride_kv[pool_kv_stride[i][0]] = pool_kv_stride[i][1:]
            if pool_kvq_kernel is not None:
                pool_kv[pool_kv_stride[i][0]] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in pool_kv_stride[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.norm_stem = norm_layer(embed_dim) if norm_stem else None

        input_size = self.patch_dims

        # not reverse
        self.blocks = nn.ModuleList()

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if dim_mul_in_att:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i])
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1])
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=False,
                rel_pos_spatial=self.rel_pos_spatial,
                rel_pos_temporal=self.rel_pos_temporal,
                rel_pos_zero_init=rel_pos_zero_init,
                residual_pooling=residual_pooling,
                dim_mul_in_att=dim_mul_in_att,
                separate_qkv=separate_qkv,
            )

            self.blocks.append(attention_block)
            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride
                    for size, stride in zip(input_size, stride_q[i])
                ]

            embed_dim = dim_out

        self.norm = norm_layer(embed_dim)

        # no detection
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embed_dim, num_classes, bias=True)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on
                    )
                    self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.fc.weight.data.mul_(head_init_scale)
        self.fc.bias.data.mul_(head_init_scale)

        # Load pretrained model
        if weights != '':
            self.load_pretrained(weights, transfer_weights, remove_fc, model=weakref.proxy(self))

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        # Update ret dict with other configs
        # Will overwrite existing keys in ret
        assert cfg.TRANSFORM.CROP_SIZE == cfg.TRANSFORM.TEST_CROP_SIZE
        arch_cfgs = cfg.MODEL.ARCH_CFGS
        act_checkpoint = True if len(arch_cfgs) > 0 and arch_cfgs[0] == 'act_checkpoint' else False
        ret.update({
            'temporal_size': cfg.DATALOADER.NUM_SEGMENTS * cfg.DATALOADER.CLIP_LENGTH,
            'spatial_size': cfg.TRANSFORM.CROP_SIZE,
            'in_chans': 3 if not cfg.TRANSFORM.USE_FLOW else 2,
            'drop_path_rate': cfg.MODEL.DROP_PATH_RATIO,
            'dropout_rate': cfg.MODEL.DROPOUT_RATIO,
            'act_checkpoint': act_checkpoint
        })
        return ret

    @staticmethod
    def transfer_weights(pre_train_dict, model=None, *args, **kwargs):
        model_dict = model.state_dict()
        if 'model_state' in pre_train_dict.keys():
            #print(pre_train_dict.keys())
            pre_train_dict = pre_train_dict['model_state']

        if 'head.projection.weight' in pre_train_dict:
            pre_train_dict['fc.weight'] = pre_train_dict['head.projection.weight']
        if 'head.projection.bias' in pre_train_dict:
            pre_train_dict['fc.bias'] = pre_train_dict['head.projection.bias']
        if "patch_embed.proj.weight" in pre_train_dict.keys() and "patch_embed.proj.weight" in model_dict.keys():
            if (
                    len(pre_train_dict["patch_embed.proj.weight"].shape)
                    == 4
                    and len(model_dict["patch_embed.proj.weight"].shape)
                    == 5
            ):  # img->video
                o_ch, i_ch, t, h, w = model_dict["patch_embed.proj.weight"].shape
                assert (o_ch, i_ch, h, w) == pre_train_dict["patch_embed.proj.weight"].shape
                prev = t // 2
                futu = t - prev - 1
                pre_train_dict[
                    "patch_embed.proj.weight"
                ] = torch.cat((
                    torch.zeros(o_ch, i_ch, prev, h, w, dtype=pre_train_dict["patch_embed.proj.weight"].dtype),
                    pre_train_dict["patch_embed.proj.weight"].unsqueeze(2),
                    torch.zeros(o_ch, i_ch, futu, h, w, dtype=pre_train_dict["patch_embed.proj.weight"].dtype)
                ), dim=2)
                logger.info(
                    f"inflate patch_embed.proj.weight to {pre_train_dict['patch_embed.proj.weight'].shape}"
                )

        qkv = [
            "attn.pool_k.weight",
            "attn.pool_q.weight",
            "attn.pool_v.weight",
        ]
        for k in pre_train_dict.keys():
            if (
                    any([x in k for x in qkv])
                    and pre_train_dict[k].shape != model_dict[k].shape
            ):
                # print(pre_train_dict[k].shape, model_dict[k].shape)
                logger.info(
                    f"inflate {k} from {pre_train_dict[k].shape} to {model_dict[k].shape}"
                )
                o_ch, i_ch, t, h, w = model_dict[k].shape
                assert (o_ch, i_ch, h, w) == pre_train_dict[k].shape
                prev = t // 2
                futu = t - prev - 1
                pre_train_dict[k] = torch.cat((
                    torch.zeros(o_ch, i_ch, prev, h, w, dtype=pre_train_dict[k].dtype),
                    pre_train_dict[k].unsqueeze(2),
                    torch.zeros(o_ch, i_ch, futu, h, w, dtype=pre_train_dict[k].dtype)
                ), dim=2)
                pre_train_dict[k].repeat(
                    1, 1, t, 1, 1
                )

        for k in pre_train_dict.keys():
            if 'rel_pos' in k and pre_train_dict[k].shape != model_dict[k].shape:
                # print(pre_train_dict[k].shape, model_dict[k].shape)
                logger.info(f"interpolating {k} from {pre_train_dict[k].shape} to {model_dict[k].shape}")
                new_pos_embed = torch.nn.functional.interpolate(
                    pre_train_dict[k].reshape(1, pre_train_dict[k].shape[0], -1).permute(0, 2, 1),
                    size=model_dict[k].shape[0],
                    mode='linear'
                )
                new_pos_embed = new_pos_embed.reshape(-1, model_dict[k].shape[0]).permute(1, 0).squeeze()
                pre_train_dict[k] = new_pos_embed

        # Match pre-trained weights that have same shape as current model.
        pre_train_dict_match = {}
        not_used_layers = []
        for k, v in pre_train_dict.items():
            if k in model_dict:
                if v.size() == model_dict[k].size():
                    pre_train_dict_match[k] = v
                else:
                    if "attn.rel_pos" in k:
                        v_shape = v.shape
                        v = v.t().unsqueeze(0)
                        v = torch.nn.functional.interpolate(
                            v,
                            size=model_dict[k].size()[0],
                            mode="linear",
                        )
                        v = v[0].t()
                        pre_train_dict_match[k] = v
                        logger.info(
                            "{} reshaped from {} to {}".format(
                                k, v_shape, v.shape
                            )
                        )
                    elif "pos_embed_temporal" in k:
                        v_shape = v.shape
                        v = torch.nn.functional.interpolate(
                            v.permute(0, 2, 1),
                            size=model_dict[k].shape[1],
                            mode="linear",
                        )
                        pre_train_dict_match[k] = v.permute(0, 2, 1)
                        logger.info(
                            "{} reshaped from {} to {}".format(
                                k, v_shape, pre_train_dict_match[k].shape
                            )
                        )
                    elif "pos_embed_spatial" in k:
                        v_shape = v.shape
                        pretrain_size = int(math.sqrt(v_shape[1]))
                        model_size = int(math.sqrt(model_dict[k].shape[1]))
                        assert pretrain_size * pretrain_size == v_shape[1]
                        assert (
                            model_size * model_size
                            == model_dict[k].shape[1]
                        )
                        v = torch.nn.functional.interpolate(
                            v.reshape(
                                1, pretrain_size, pretrain_size, -1
                            ).permute(0, 3, 1, 2),
                            size=(model_size, model_size),
                            mode="bicubic",
                        )
                        pre_train_dict_match[k] = v.reshape(
                            1, -1, model_size * model_size
                        ).permute(0, 2, 1)
                        logger.info(
                            "{} reshaped from {} to {}".format(
                                k, v_shape, pre_train_dict_match[k].shape
                            )
                        )
                    else:
                        not_used_layers.append(k)
            else:
                not_used_layers.append(k)
        return pre_train_dict_match

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def forward(self, x , featuremap=None):
        x = x[kfg.FRAMES]
        #print(x.shape)
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape
        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        for blk in self.blocks:
            x, thw = blk(x, thw)

        feature_map7x7 = x

        if self.use_mean_pooling:
            if self.cls_embed_on:
                x = x[:, 1:]
                x = x.mean(1)
                x = self.norm(x)
        elif self.cls_embed_on:
            x = self.norm(x)
            x = x[:, 0]
        else: # this is default, [norm -> mean]
            x = self.norm(x)
            x = x.mean(1)
        x = self.fc(x)
        #print(x.shape)
        #print(x.shape)
        # print(feature_map7x7.shape)
        if featuremap != None:
            if self.training:
                return x, feature_map7x7
            else:
                return x.softmax(dim=1),feature_map7x7
        else:
            if self.training:
                return x
            else:
                return x.softmax(dim=1)

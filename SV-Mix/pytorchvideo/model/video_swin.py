import torch
import torch.nn as nn
import numpy as np

from .build import MODEL_REGISTRY
from .base_net import BaseNet
from .swin3d_vit import SwinTransformer3D

from pytorchvideo.config import kfg, configurable
from pytorchvideo.utils.vit_helpers import to_2tuple


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
        # print(x.shape)
        # FIXME look at relaxing size constraints
        assert T == self.early_stride and H == self.img_size[0] and W == self.img_size[1],  \
            f"Input image size ({T}*{H}*{W}) doesn't match model ({self.early_stride}*{self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


@MODEL_REGISTRY.register()
class VIDEO_SWIN(SwinTransformer3D, BaseNet):
    _arch_dict = {
        'video_swin_vit_s_p4_w7': dict(drop_path_rate=0.3, window_size=(4,7,7), patch_size=(2,4,4), in_chans=3,
                                     embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
        
        'video_swin_vit_b_p4_w7': dict(drop_path_rate=0.5, window_size=(4,7,7), patch_size=(2,4,4), in_chans=3,
                                     embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
        'video_swin_vit_l_p4_w7': dict(drop_path_rate=0.5, window_size=(4,7,7), patch_size=(2,4,4), inchans=3,
                                     embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
    }

    @configurable
    def __init__(self, img_size=224, early_stride=2, patch_size=4, in_chans=3, num_classes=1000, embed_dim=768,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, dropout_ratio=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False,
                 weights='', transfer_weights=False, remove_fc=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        # print(window_size)
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, dropout_ratio=dropout_ratio,
                         norm_layer=norm_layer, ape=ape, patch_norm=patch_norm, use_checkpoint=use_checkpoint)

        self.early_stride = early_stride
        # self.patch_embed = TubeletEmbed(img_size=img_size, early_stride=early_stride, patch_size=patch_size,
        #                                 in_chans=in_chans, embed_dim=embed_dim)
        self.apply(self._init_weights)

        # Load pretrained model
        if weights != '':
            # print(weights)
            self.load_pretrained(weights, transfer_weights, remove_fc, early_stride=early_stride, current_model=self.state_dict(), window_size=self.window_size)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        # Update ret dict with other configs
        # Will overwrite existing keys in ret
        ret.update({
            'img_size': cfg.TRANSFORM.CROP_SIZE,
            'dropout_ratio': cfg.MODEL.DROPOUT_RATIO,
            'early_stride': cfg.MODEL.EARLY_STRIDE,
        })
        return ret

    @staticmethod
    def transfer_weights(state_dict, current_model=None, window_size=None, early_stride=None, *args, **kwargs):
        new_state_dict = {}
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]
            
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = current_model[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            # print('\n\n\n',L1, nH1,L2, nH2)
            L2s = (2*window_size[1]-1) * (2*window_size[2]-1)
            wd = window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2s and L1<L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*window_size[1]-1, 2*window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2s).permute(1, 0)
                    state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)
                elif L1 == L2s and L1<L2:
                    state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)
                elif L1>L2:
                    # print('\n\n\n enter')
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, 2*8-1, (2*7-1)*(2*7-1) ), size=(2*window_size[0]-1, (2*window_size[1]-1)*(2*window_size[2]-1) ),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
                    state_dict[k] = relative_position_bias_table_pretrained
                # print(state_dict[k].shape,relative_position_bias_table_current.size())
            
        
        for k, v in state_dict.items():
            # if 'swin.' in k:
            #     k=k.replace('swin.','')
            # if 'embeddings.' in k:
            #     k=k.replace('embeddings.','')
            # if 'encoder.' in k:
            #     k=k.replace('encoder.','')
            # print(k)
            v = v.detach().numpy()
            
            if k == 'patch_embed.proj.weight':
                shape = v.shape
                if len(shape) == 4:
                    v = np.reshape(v, newshape=[shape[0], shape[1], 1, shape[2], shape[3]])
                    #if early_stride != 1:
                        # if early_stride != 2:
                        #     s1 = early_stride // 2
                        #     s2 = early_stride - early_stride // 2 - 1
                        #     v = np.concatenate((np.zeros(shape=(shape[0], shape[1], s1, shape[2], shape[3])),
                        #                         v,
                        #                         np.zeros(shape=(shape[0], shape[1], s2, shape[2], shape[3]))), axis=2)
                        # else:
                    v = v/2
                    # cout,cin,h,w=v.shape
                    # print(k,v.shape)
                    # v = v[:,:,None,:,:]
                    v = np.repeat(v,2,axis=2)
                    # print(v.shape)
            new_state_dict[k] = torch.from_numpy(v)
        return new_state_dict

    def forward(self, x, featuremap=None):
        x = x[kfg.FRAMES]
        bsz = x.size(0)
        chn = x.size(1)
        lgt = x.size(2)
        hig = x.size(3)
        wid = x.size(4)
        # x = x.view(bsz, chn, lgt // self.early_stride, self.early_stride, hig, wid).transpose(1, 2)
        # x = x.reshape(bsz * lgt // self.early_stride, chn, self.early_stride, hig, wid)

        x, feature_map7x7 = self.forward_features(x, featuremap)
        # print(x.shape)
        # x = self.drop(x)
        x = self.fc(x)
        # y = x.view(bsz, lgt // self.early_stride, -1).mean(dim=1)
        # if self.training:
        #     return x
        # else:
        #     return x.softmax(dim=1)
        
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

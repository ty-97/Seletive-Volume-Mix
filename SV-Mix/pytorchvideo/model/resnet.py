import weakref
import torch.nn as nn
from .build import MODEL_REGISTRY
from .base_net import BaseNet
from pytorchvideo.config import kfg, configurable


__all__ = ['ResNet']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@MODEL_REGISTRY.register()
class ResNet(BaseNet):
    _arch_dict = {
        # ResNet model from
        # `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        'resnet18': dict(block=BasicBlock, layers=[2, 2, 2, 2]),
        'resnet34': dict(block=BasicBlock, layers=[3, 4, 6, 3]),
        'resnet50': dict(block=Bottleneck, layers=[3, 4, 6, 3]),
        'resnet101': dict(block=Bottleneck, layers=[3, 4, 23, 3]),
        'resnet103': dict(block=Bottleneck, layers=[3, 4, 23, 3], deep_stem=True),
        'resnet152': dict(block=Bottleneck, layers=[3, 8, 36, 3]),

        # ResNeXt model from
        # `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
        'resnext50_32x4d': dict(block=Bottleneck, layers=[3, 4, 6, 3], groups=32, width_per_group=4),
        'resnext101_32x8d': dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=8),

        # Wide ResNet model from
        # `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
        #
        # The model is the same as ResNet except for the bottleneck number of channels
        # which is twice larger in every block. The number of channels in outer 1x1
        # convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
        # channels, and in Wide ResNet-50-2 has 2048-1024-2048.
        'wide_resnet50_2': dict(block=Bottleneck, layers=[3, 4, 6, 3], width_per_group=64 * 2),
        'wide_resnet101_2': dict(block=Bottleneck, layers=[3, 4, 23, 3], width_per_group=64 * 2)
    }

    @configurable
    def __init__(self, block, layers, num_classes=1000, dropout_ratio=0.5, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, deep_stem=False,
                 weights='', remove_fc=False, frozen_bn=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self._deep_stem = deep_stem

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if not deep_stem:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = norm_layer(self.inplanes)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(self.inplanes, self.inplanes * 2, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = norm_layer(self.inplanes * 2)
            self.relu3 = nn.ReLU(inplace=True)
            
            self.inplanes *= 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.drop = nn.Dropout(dropout_ratio)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # Load pretrained model
        if weights != '':
            self.load_pretrained(weights, False, remove_fc)

        # Frozen BN layers
        if frozen_bn:
            self.frozen_bn(weakref.proxy(self))

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.pop('transfer_weights', None)

        # Update ret dict with other configs
        # Will overwrite existing keys in ret
        ret.update({
            'dropout_ratio': cfg.MODEL.DROPOUT_RATIO,
            'frozen_bn': cfg.MODEL.FROZEN_BN
        })
        return ret

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, layer, temporal_pool=True):
        bsz = x.size(0)
        chn = x.size(1)
        hig = x.size(2)
        wid = x.size(3)

        # See note [TorchScript super()]
        if not self._deep_stem:
            x = self.conv1(x.view(bsz * chn // 3, 3, hig, wid))
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x.view(bsz * chn // 3, 3, hig, wid))
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if layer == 5:
            return x
        x_g = self.pool(x).flatten(1)
        if layer == 6:
            return x_g

        x_g = self.drop(x_g)

        x1 = self.fc(x_g)
        if temporal_pool:
            return x1.view(bsz, chn // 3, -1).mean(dim=1)
        else:
            return x1.view(bsz, chn // 3, -1)

    def forward(self, x, layer=7):
        x = x[kfg.FRAMES]
        y = self._forward_impl(x, layer)
        if self.training:
            return y
        else:
            return y.softmax(dim=1)

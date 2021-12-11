import copy
import torch
import warnings
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models.backbones import HRNet
from mmdet.models import BACKBONES, build_backbone

@BACKBONES.register_module()
class HRNet3D(HRNet):
    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=False,
                 multiscale_output=True,
                 pretrained=None,
                 init_cfg=None):
        super(HRNet3D, self).__init__(extra, in_channels, conv_cfg, norm_cfg, norm_eval, with_cp, zero_init_residual, pretrained, init_cfg)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

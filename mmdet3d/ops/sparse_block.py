from mmcv.cnn import build_conv_layer, build_norm_layer
from torch import nn

from .spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseModule, SparseSequential
else:
    from mmcv.ops import SparseModule, SparseSequential

from mmdet.models.backbones.resnet import BasicBlock, Bottleneck

def replace_feature(out, new_features):
    if 'replace_feature' in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class SparseBottleneck(Bottleneck, SparseModule):
    """Sparse bottleneck block for PartA^2.

    Bottleneck block implemented with submanifold sparse convolution.

    Args:
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        stride (int): stride of the first block. Default: 1
        downsample (None | Module): down sample module for block.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=None):

        SparseModule.__init__(self)
        Bottleneck.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv3(out)
        out = replace_feature(out, self.bn3(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class SparseBasicBlock(BasicBlock, SparseModule):
    """Sparse basic block for PartA^2.

    Sparse basic block implemented with submanifold sparse convolution.

    Args:
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        stride (int): stride of the first block. Default: 1
        downsample (None | Module): down sample module for block.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_type='relu'):
        SparseModule.__init__(self)
        BasicBlock.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        act_type = act_type.lower()
        # a confused way
        if act_type != 'relu':
            if act_type == 'gelu':
                self.relu = nn.GELU()
            elif act_type == 'silu':
                self.relu = nn.SiLU(inplace=True)


    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'

        out = self.conv1(x)
        out = replace_feature(out, self.norm1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

class AdaptiveSparseBasicBlock(BasicBlock, SparseModule):
    """
    Adaptive stride and channels
    """

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 merge=True,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=None):
        SparseModule.__init__(self)
        BasicBlock.__init__(
            self,
            planes,
            planes,
            stride=1,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        if isinstance(stride, (tuple, list)):
            is_stride = max(stride) > 1
        else:
            is_stride = stride > 1

        if inplanes != planes or is_stride:
            ndim = int(conv_cfg['type'][-2])
            assert ndim in (1, 2, 3, 4)
            ada_conv_cfg = {}
            ada_conv_cfg['type'] = f'SparseConv{ndim}d'
            ada_conv_cfg['indice_key'] = conv_cfg['indice_key'] + '.adaptive'
            if merge:
                self.ada_conv = build_conv_layer(ada_conv_cfg, inplanes, planes, stride, stride=stride, padding=0)
            else:
                self.ada_conv = build_conv_layer(ada_conv_cfg, inplanes, planes, 3, stride=stride, padding=1)
            self.ada_norm = build_norm_layer(norm_cfg, planes)[1]
            self.ada_relu = nn.ReLU(inplace=True)

    def forward(self, x):

        if hasattr(self, 'ada_conv'):
            x = self.ada_conv(x)
            x = replace_feature(x, self.ada_norm(x.features))
            x = replace_feature(x, self.ada_relu(x.features))

        identity = x.features

        assert x.features.dim() == 2, f'x.features.dim()={x.features.dim()}'

        out = self.conv1(x)
        out = replace_feature(out, self.norm1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


def make_sparse_convmodule(in_channels,
                           out_channels,
                           kernel_size,
                           indice_key,
                           stride=1,
                           padding=0,
                           conv_type='SubMConv3d',
                           act_type='relu',
                           norm_cfg=None,
                           order=('conv', 'norm', 'act')):
    """Make sparse convolution module.

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of out channels
        kernel_size (int|tuple(int)): kernel size of convolution
        indice_key (str): the indice key used for sparse tensor
        stride (int|tuple(int)): the stride of convolution
        padding (int or list[int]): the padding number of input
        conv_type (str): sparse conv type in spconv
        norm_cfg (dict[str]): config of normalization layer
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").

    Returns:
        spconv.SparseSequential: sparse convolution module.
    """
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {'conv', 'norm', 'act'} == {'conv', 'norm', 'act'}

    conv_cfg = dict(type=conv_type, indice_key=indice_key)

    layers = list()
    for layer in order:
        if layer == 'conv':
            if conv_type not in [
                    'SparseInverseConv4d',
                    'SparseInverseConv3d',
                    'SparseInverseConv2d',
                    'SparseInverseConv1d'
            ]:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False))
            else:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        bias=False))
        elif layer == 'norm':
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == 'act':
            act_type = act_type.lower()
            if act_type == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif act_type == 'gelu':
                layers.append(nn.GELU())
            elif act_type == 'silu':
                layers.append(nn.SiLU(inplace=True))
            else:
                raise NotImplementedError


    layers = SparseSequential(*layers)
    return layers
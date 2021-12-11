import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mmcv.runner import auto_fp16
from mmcv.cnn import build_norm_layer

from mmdet3d.ops import flat2window, window2flat, SRATensor, DebugSRATensor, spconv

from ipdb import set_trace
import os
import pickle as pkl

class WindowAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, batch_first=False, layer_id=None):
        super().__init__()
        self.nhead = nhead

        # from mmdet3d.models.transformer.my_multi_head_attention import MyMultiheadAttention
        # self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.layer_id = layer_id

    def forward(self, sra_tensor, do_shift):
        '''
        Args:

        Out:
            shifted_feat_dict: the same type as window_feat_dict
        '''
        assert isinstance(sra_tensor, SRATensor)

        out_feat_dict = {}
        win_feat_dict, mask_dict = sra_tensor.window_tensor(do_shift)
        pos_dict = sra_tensor.position_embedding(do_shift)

        for name in win_feat_dict:
            #  [n, num_token, embed_dim]
            pos = pos_dict[name]
            pos = pos.permute(1, 0, 2)

            feat_3d = win_feat_dict[name]
            feat_3d = feat_3d.permute(1, 0, 2)

            key_padding_mask = mask_dict[name]

            v = feat_3d
            q = k = feat_3d + pos

            out_feat_3d, attn_map = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)
            out_feat_dict[name] = out_feat_3d.permute(1, 0, 2)

        sra_tensor.update(out_feat_dict)
        
        return sra_tensor

class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, layer_id=None, mlp_dropout=0):
        super().__init__()
        assert not batch_first, 'Current version of PyTorch does not support batch_first in MultiheadAttention. After upgrading pytorch, do not forget to check the layout of MLP and layer norm to enable batch_first option.'
        self.batch_first = batch_first
        self.win_attn = WindowAttention(d_model, nhead, dropout, layer_id=layer_id)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(mlp_dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)

        self.activation = _get_activation_fn(activation)
        self.fp16_enabled=True

    # @auto_fp16(apply_to=('att_input'))
    def forward(
        self,
        input,
        do_shift
        ):
        assert isinstance(input, SRATensor)
        src = input.features
        output = self.win_attn(input, do_shift) #[N, d_model]
        src2 = output.features
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        output.set_features(src)
        # output = spconv.SparseConvTensor(src, att_input.indices, None, None)
        # att_output.features = src

        return output

class SRABlock(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''

    def __init__(self, key, d_model, nhead, dim_feedforward, window_shape, dropout=0.1,
                 activation="relu", batch_first=False, block_id=-100):
        super().__init__()
                # SRABlock(d_model[i], nhead[i], dim_feedforward[i], window_shape
                #     dropout, activation, batch_first=False, block_id=i)

        encoder_1 = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
            activation, batch_first, layer_id=block_id * 2 + 0)
        encoder_2 = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
            activation, batch_first, layer_id=block_id * 2 + 1)
        # BasicShiftBlock(d_model[i], nhead[i], dim_feedforward[i], dropout, activation, batch_first=False)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])
        self.window_shape = window_shape
        self.key = key

    def forward(
        self,
        input,
        batching_info,
        using_checkpoint=False,
        ):
        assert isinstance(input, SRATensor)

        output = input
        if not output.ready:
            output.setup(batching_info, self.key, self.window_shape, 10000)
        for i in range(2):

            layer = self.encoder_list[i]
            if using_checkpoint:
                output = checkpoint(layer, output, i == 1)
            else:
                output = layer(output, i == 1)

        return output

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
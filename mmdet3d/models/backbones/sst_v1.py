from mmdet.models import BACKBONES

import torch
import torch.nn as nn
import copy
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d.models.sst.sst_basic_block import BasicShiftBlock

from mmdet3d.ops import flat2window, window2flat
from ipdb import set_trace


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@BACKBONES.register_module()
class SSTv1(nn.Module):
    '''
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checckpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    '''

    def __init__(
        self,
        d_model=[],
        nhead=[],
        num_blocks=6,
        dim_feedforward=[],
        dropout=0.0,
        activation="gelu",
        output_shape=None,
        num_attached_conv=2,
        conv_in_channel=64,
        conv_out_channel=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        debug=True,
        drop_info=None,
        normalize_pos=False,
        pos_temperature=10000,
        window_shape=None,
        in_channel=None,
        conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1),
        checkpoint_blocks=[],
        ):
        super().__init__()
        
        assert drop_info is not None
        self.meta_drop_info = drop_info
        self.pos_temperature = pos_temperature
        self.d_model = d_model
        self.window_shape = window_shape
        self.normalize_pos = normalize_pos
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks

        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])

        # Sparse Regional Attention Blocks
        block_list=[]
        for i in range(num_blocks):
            block_list.append(
                BasicShiftBlock(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i)
            )

        self.block_list = nn.ModuleList(block_list)
            
        self._reset_parameters()

        self.output_shape = output_shape

        self.debug = debug

        self.num_attached_conv = num_attached_conv

        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):

                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]

                if i > 0:
                    conv_in_channel = conv_out_channel
                conv = build_conv_layer(
                    conv_cfg,
                    in_channels=conv_in_channel,
                    out_channels=conv_out_channel,
                    **conv_kwargs_i,
                    )

                if norm_cfg is None:
                    convnormrelu = nn.Sequential(
                        conv,
                        nn.ReLU(inplace=True)
                    )
                else:
                    convnormrelu = nn.Sequential(
                        conv,
                        build_norm_layer(norm_cfg, conv_out_channel)[1],
                        nn.ReLU(inplace=True)
                    )
                conv_list.append(convnormrelu)
            
            self.conv_layer = nn.ModuleList(conv_list)

    def forward(self, input_tuple):
        '''
        '''
        voxel_feat, ind_dict_list, voxel_info = input_tuple # 3 outputs of SSTInputLayer, containing pre-computed information for Sparse Regional Attention.

        assert voxel_info['coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'
        self.set_drop_info()
        device = voxel_info['coors'].device
        batch_size = voxel_info['coors'][:, 0].max().item() + 1

        num_shifts = len(ind_dict_list) # Usually num_shifts == 2, one for non-shifted layout, one for shifted layout
        
        padding_mask_list = [
            self.get_key_padding_mask(ind_dict_list[i], voxel_info[f'voxel_drop_level_shift{i}'], device) 
            for i in range(num_shifts)
        ]
        pos_embed_list = [
            self.get_pos_embed(
                _t,
                voxel_info[f'coors_in_win_shift{i}'],
                voxel_info[f'voxel_drop_level_shift{i}'],
                voxel_feat.dtype,
                voxel_info.get(f'voxel_win_level_shift{i}', None)
            ) 
            for i, _t in enumerate(ind_dict_list) # 2-times for-loop, one for non-shifted layout, one for shifted layout
        ]
        voxel_drop_level_list = [voxel_info[f'voxel_drop_level_shift{i}'] for i in range(num_shifts)]

        output = voxel_feat
        if hasattr(self, 'linear0'):
            output = self.linear0(output)
        for i, block in enumerate(self.block_list):
            output = block(output, pos_embed_list, ind_dict_list, voxel_drop_level_list,
                padding_mask_list, self.drop_info, using_checkpoint = i in self.checkpoint_blocks)
        
        output = self.recover_bev(output, voxel_info['coors'], batch_size)

        if self.num_attached_conv > 0:
            for conv in self.conv_layer:
                output = conv(output)

        output_list = []
        output_list.append(output)

        return output_list
    
    def get_key_padding_mask(self, ind_dict, voxel_drop_lvl, device):
        num_all_voxel = len(voxel_drop_lvl)
        key_padding = torch.ones((num_all_voxel, 1)).to(device).bool()

        window_key_padding_dict = flat2window(key_padding, voxel_drop_lvl, ind_dict, self.drop_info)

        # logical not. True means masked
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)
        
        return window_key_padding_dict
        
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas

    def get_pos_embed(self, ind_dict, coors_in_win, voxel_drop_level, dtype, voxel_window_level):
        '''
        Args:
        '''

        # [N,]
        win_x, win_y = self.window_shape

        x, y = coors_in_win[:, 0] - win_x/2, coors_in_win[:, 1] - win_y/2
        assert (x >= -win_x/2 - 1e-4).all()
        assert (x <= win_x/2-1 + 1e-4).all()

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
        
        pos_length = self.d_model[0] // 2
        assert self.d_model[0] == self.d_model[1] == self.d_model[-1], 'If you want to use different d_model, Please implement corresponding pos embendding.'
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=coors_in_win.device)
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()],
                              dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()],
                              dim=-1).flatten(1)
        # [num_tokens, pos_length * 2]
        pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1).to(dtype)

        window_pos_emb_dict = flat2window(
            pos_embed_2d, voxel_drop_level, ind_dict, self.drop_info)
        
        return window_pos_emb_dict

    def set_drop_info(self):
        if hasattr(self, 'drop_info'):
            return
        meta = self.meta_drop_info
        if isinstance(meta, tuple):
            if self.training:
                self.drop_info = meta[0]
            else:
                self.drop_info = meta[1]
        else:
            self.drop_info = meta
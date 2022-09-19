# Do not use this file. Please wait for future release.
from mmdet.models import BACKBONES

import torch
import torch.nn as nn
import copy
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d.models.sst.sra_block import SRABlock

from ipdb import set_trace


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@BACKBONES.register_module()
class SST(nn.Module):

    def __init__(
        self,
        d_model=[],
        nhead=[],
        num_blocks=6,
        dim_feedforward=[],
        dropout=0.1,
        activation="relu",
        output_shape=None,
        num_attached_conv=0,
        conv_in_channel=64,
        conv_out_channel=64,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        debug=True,
        batch_first=False,
        batching_info=None,
        no_pos_embed=False,
        normalize_pos=False,
        pos_temperature=10000,
        window_shape=None,
        init_sparse_shape=None,
        in_channel=None,
        conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1),
        checkpoint_blocks=[],
        key='single_unique_key',
        fp16=True,
        ):
        super().__init__()
        
        assert isinstance(batching_info, tuple)
        self.batching_info = batching_info
        self.no_pos_embed = no_pos_embed
        self.pos_temperature = pos_temperature
        self.d_model = d_model
        self.window_shape = window_shape
        self.key = key
        self.normalize_pos = normalize_pos
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks
        self.init_sparse_shape = init_sparse_shape
        self.fp16 = fp16

        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])

        block_list=[]
        for i in range(num_blocks):
            block_list.append(
                SRABlock(
                    key, d_model[i], nhead[i], dim_feedforward[i], window_shape,
                    dropout, activation, batch_first=False, block_id=i
                )
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
        Note that, batch_first is set to True
        Args:
        feat_3d_list: list[Tensor of shape(bs, max_num_token, embed_dim)]
        Outs:
        output: list[Tensor of shape: (bs, embed_dim, h, w)]
                output tensor is in bev view
        '''
        voxel_feats, voxel_coors, batch_size = input_tuple
        voxel_coors = voxel_coors.long()
        if self.fp16:
            voxel_feats = voxel_feats.to(torch.half)
        if self.training:
            batching_info = self.batching_info[0]
        else:
            batching_info = self.batching_info[1]

        device = voxel_feats.device
        
        if hasattr(self, 'linear0'):
            voxel_feats = self.linear0(voxel_feats)
        
        output = SRATensor(voxel_feats, voxel_coors, self.init_sparse_shape, batch_size)
        # output = DebugSRATensor(voxel_feats, voxel_coors, self.init_sparse_shape, batch_size)
        # output = spconv.SparseConvTensor(voxel_feats, voxel_coors, self.init_sparse_shape, batch_size)

        for i, block in enumerate(self.block_list):
            output = block(output, batching_info, using_checkpoint = i in self.checkpoint_blocks)
        
        # to bev
        output = self._window2bev_old(output.features, output.indices, batch_size)
        # output = self._window2bev_old(output, voxel_coors, batch_size)

        if self.num_attached_conv > 0:
            for conv in self.conv_layer:
                output = conv(output)

        output_list = []
        output_list.append(output)

        return output_list
        
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def _window2bev_old(self, voxel_feat, coors, batch_size):
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

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas
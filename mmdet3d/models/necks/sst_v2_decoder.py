from mmdet.models import NECKS

import torch
import torch.nn as nn
import copy
from mmcv.cnn import build_conv_layer, build_norm_layer

from mmdet3d.models.backbones import SSTv2
from mmdet3d.models.sst.sst_basic_block_v2 import BasicShiftBlockV2

from mmdet3d.ops import flat2window_v2, window2flat_v2


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@NECKS.register_module()
class SSTv2Decoder(SSTv2):
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
        debug=True,
        in_channel=None,
        checkpoint_blocks=[],
        layer_cfg=dict(),
        use_fake_voxels=True,
        ):

        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_blocks=num_blocks,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            output_shape=output_shape,
            num_attached_conv=0,
            debug=debug,
            checkpoint_blocks=checkpoint_blocks,
            layer_cfg=layer_cfg,
            masked=True,)

        if in_channel is not None:
            self.enc2dec_projection = nn.Linear(in_channel, d_model[0])
        self._reset_parameters()

        self.use_fake_voxels = use_fake_voxels
        self.mask_token = nn.Parameter(torch.zeros(1, d_model[0]))
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, voxel_info):
        '''
        '''
        voxel_info_encoder = voxel_info
        voxel_info_decoder = voxel_info["voxel_info_decoder"]

        # _____ add in encoder output to the input _____
        encoder_out = voxel_info_encoder["output"]

        # if in_channel project encoder output to right dimension
        if hasattr(self, 'enc2dec_projection'):
            encoder_out = self.enc2dec_projection(encoder_out)

        # replace unmasked voxels with encoder value
        dec2enc_idx = voxel_info_decoder["dec2enc_idx"]
        voxel_feat = voxel_info_decoder['voxel_feats']
        voxel_feat[dec2enc_idx] = encoder_out
        assert torch.allclose(voxel_info_decoder['voxel_coors'][dec2enc_idx], voxel_info_encoder["voxel_coors"]), \
            "Mapping dec2enc not valid"

        # replace masked voxels with masking token
        dec2masked_idx = voxel_info_decoder["dec2masked_idx"]
        n_masked = voxel_info_decoder["n_masked"]
        masked_tokens = self.mask_token.repeat(n_masked, 1)
        voxel_feat[dec2masked_idx] = masked_tokens
        if self.use_fake_voxels:
            # replace fake voxels with masking token
            dec2fake_idx = voxel_info_decoder["dec2fake_idx"]
            n_fake = voxel_info_decoder["n_fake"]
            masked_tokens = self.mask_token.repeat(n_fake, 1)
            voxel_feat[dec2fake_idx] = masked_tokens
        voxel_info_decoder['voxel_feats'] = voxel_feat

        if self.debug:
            test_mapping = -torch.ones(len(voxel_feat), device=voxel_feat.device)
            test_mapping[dec2enc_idx] = 0
            test_mapping[dec2masked_idx] = 1
            if self.use_fake_voxels:
                test_mapping[dec2fake_idx] = 0
            assert not (test_mapping == -1).any(), "All voxels are not covered by the enc_idx and masked_idx"
            assert test_mapping.sum() == n_masked, \
                f"The number of masked voxels differ {test_mapping.sum()} vs {n_masked}"
            assert (1-test_mapping).sum() == len(voxel_feat)-n_masked, \
                f"The number of unmasked voxels differ  {(1-test_mapping).sum()} vs {len(voxel_feat)-n_masked}"
            assert (test_mapping[voxel_info_decoder["dec2input_idx"]].long() == voxel_info_decoder["mask"].long()
                    ).all(), "The masking of the mismatches"

        voxel_info_decoder = super().forward(voxel_info_decoder)

        return voxel_info, voxel_info_decoder, voxel_info_encoder

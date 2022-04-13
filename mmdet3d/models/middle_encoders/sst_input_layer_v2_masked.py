# move the computation of position embeding and mask in middle_encoder_layer
import math
import numpy as np

import torch
from mmcv.runner import auto_fp16
from torch import nn

from . import SSTInputLayerV2
from ..builder import MIDDLE_ENCODERS
from mmdet3d.ops import flat2window_v2, window2flat_v2, get_inner_win_inds, make_continuous_inds, get_flat2win_inds_v2, get_window_coors
import random
import pickle as pkl
import os

@MIDDLE_ENCODERS.register_module()
class SSTInputLayerV2Masked(SSTInputLayerV2):
    """
    This is one of the core class of SST, converting the output of voxel_encoder to sst input.
    There are 3 things to be done in this class:
    1. Reginal Grouping : assign window indices to each voxel.
    2. Voxel drop and region batching: see our paper for detail
    3. Pre-computing the transfomation information for converting flat features ([N x C]) to region features ([R, T, C]).
        R is the number of regions containing at most T tokens (voxels). See function flat2window and window2flat for details.

    Main args:
        drop_info (dict): drop configuration for region batching. 
        window_shape (tuple[int]): (num_x, num_y). Each window is divided to num_x * num_y pillars (including empty pillars).
        shift_list (list[tuple]): [(shift_x, shift_y), ]. shift_x = 5 means all windonws will be shifted for 5 voxels along positive direction of x-aixs.
        debug: apply strong assertion for developing. 
    """

    def __init__(self,
        drop_info,
        window_shape,
        sparse_shape,
        shuffle_voxels=True,
        debug=True,
        normalize_pos=False,
        pos_temperature=10000,
        mute=False,
        masking_ratio=0.7
        ):
        super().__init__(
            drop_info,
            window_shape,
            sparse_shape,
            shuffle_voxels=shuffle_voxels,
            debug=debug,
            normalize_pos=normalize_pos,
            pos_temperature=pos_temperature,
            mute=mute,
        )
        self.masking_ratio = 0.7

    @auto_fp16(apply_to=('voxel_feat', ))
    def forward(self, voxel_feats, voxel_coors, low_level_point_feature, indices, batch_size=None):
        '''
        Args:
            voxel_feats: shape=[N, C], N is the voxel num in the batch.
            coors: shape=[N, 4], [b, z, y, x]
            low_level_point_feature: shape=[Np, 10], Np is the point num in the batch.
            indices: shape=[Np], point voxel index
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        '''
        # TODO: Potentially add fake voxels
        gt_dict = {}

        batch_size = voxel_coors[:, 0].max() + 1
        vx, vy, vz = self.sparse_shape
        max_index = batch_size.long().item()*vz*vy*vx
        tmp = torch.bincount(indices.long())
        non_zero_voxels = torch.where(tmp)
        n_points_per_voxel = torch.zeros(max_index, device=voxel_feats.device, dtype=torch.long)
        n_points_per_voxel[non_zero_voxels] = tmp[non_zero_voxels].long()

        input_index = (
            voxel_coors[:, 0] * vz * vy * vx +  # batch
            voxel_coors[:, 1] * vy * vx +  # z
            voxel_coors[:, 2] * vx +  # y
            voxel_coors[:, 3]  # x
        )
        n_points_per_voxel = n_points_per_voxel[input_index]

        gt_dict["num_points_per_voxel"] = n_points_per_voxel

        n_unmasked_voxels = int(len(voxel_feats)*self.masking_ratio)
        mask = torch.ones(len(voxel_feats), device=voxel_feats.device)
        mask[:n_unmasked_voxels] = 0
        shuffle_inds = torch.randperm(len(voxel_feats))

        mask = mask[shuffle_inds]
        unmasked_idx = shuffle_inds[:n_unmasked_voxels]
        masked_idx = shuffle_inds[n_unmasked_voxels:]

        unmasked_voxels = voxel_feats[unmasked_idx]
        unmasked_voxel_coors = voxel_coors[unmasked_idx]

        # Might drop voxels
        voxel_info_decoder = super().forward(voxel_feats, voxel_coors, batch_size=None)
        assert len(voxel_info_decoder["voxel_feats"]) == len(voxel_feats), "Dropping is not allowed for reconstruction"

        voxel_info_decoder["n_unmasked"] = n_unmasked_voxels
        voxel_info_decoder["n_masked"] = len(voxel_feats) - n_unmasked_voxels
        voxel_info_decoder["mask"] = mask
        voxel_info_decoder["unmasked_idx"] = unmasked_idx
        voxel_info_decoder["masked_idx"] = masked_idx

        voxel_info_encoder = super().forward(unmasked_voxels, unmasked_voxel_coors, batch_size=None)
        assert len(voxel_info_encoder["voxel_feats"]) == n_unmasked_voxels, "Dropping is not allowed for reconstruction"

        dec2dec_input_idx = torch.argsort(voxel_info_decoder["original_index"])
        dec2masked_idx = dec2dec_input_idx[masked_idx]
        dec2unmasked_idx = dec2dec_input_idx[unmasked_idx]
        dec2enc_idx = dec2unmasked_idx[voxel_info_encoder["original_index"]]
        voxel_info_decoder["dec2input_idx"] = dec2dec_input_idx

        voxel_info_decoder["dec2unmasked_idx"] = dec2unmasked_idx
        voxel_info_decoder["dec2masked_idx"] = dec2masked_idx
        voxel_info_decoder["dec2enc_idx"] = dec2enc_idx

        # Debug - sanity check
        decoder_feats = voxel_info_decoder["voxel_feats"]
        decoder_coors = voxel_info_decoder["voxel_coors"]

        encoder_feats = voxel_info_encoder["voxel_feats"]
        encoder_coors = voxel_info_encoder["voxel_coors"]

        assert torch.allclose(decoder_feats[dec2dec_input_idx], voxel_feats), \
            "The mapping from decoder to decoder input is invalid"
        assert torch.allclose(decoder_coors[dec2dec_input_idx], voxel_coors.long()), \
            "The mapping from decoder to decoder input is invalid"

        assert torch.allclose(decoder_feats[dec2masked_idx], voxel_feats[masked_idx]), \
            "The mapping from decoder to masked input is invalid"
        assert torch.allclose(decoder_coors[dec2masked_idx], voxel_coors[masked_idx].long()), \
            "The mapping from decoder to masked input is invalid"

        assert torch.allclose(decoder_feats[dec2unmasked_idx], unmasked_voxels), \
            "The mapping from decoder to encoder input is invalid"
        assert torch.allclose(decoder_coors[dec2unmasked_idx], unmasked_voxel_coors.long()), \
            "The mapping from decoder to encoder input is invalid"

        assert torch.allclose(decoder_feats[dec2enc_idx], encoder_feats), \
            "The mapping from decoder to encoder output is invalid"
        assert torch.allclose(decoder_coors[dec2enc_idx], encoder_coors.long()), \
            "The mapping from decoder to encoder output is invalid"

        voxel_info_decoder["gt_dict"] = gt_dict
        voxel_info_encoder["voxel_info_decoder"] = voxel_info_decoder

        return voxel_info_encoder

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
        print(f'drop_info is set to {self.drop_info}, in input_layer')
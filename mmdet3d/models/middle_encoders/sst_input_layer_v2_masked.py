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
        voxel_size,
        shuffle_voxels=True,
        debug=True,
        normalize_pos=False,
        pos_temperature=10000,
        mute=False,
        masking_ratio=0.7,
        drop_points_th=100,
        pred_dims=3
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
        self.masking_ratio = masking_ratio
        self.drop_points_th = drop_points_th
        self.pred_dims = pred_dims
        self.voxel_size = voxel_size

    @auto_fp16(apply_to=('voxel_feat', ))
    def forward(self, voxel_feats, voxel_coors, low_level_point_feature, point_coors, batch_size=None):
        '''
        Args:
            voxel_feats: shape=[N, C], N is the voxel num in the batch.
            voxel_coors: shape=[N, 4], [b, z, y, x], voxel coordinate for each voxel
            low_level_point_feature: shape=[Np, 10] [x, y, z, I, cl_x, cl_y, cl_z, ce_x, ce_y, ce_z],
             Np is the point num in the batch, cl_* and ce_*  is position relative to the cluster and center resp.
            point_coors: shape=[Np, 4], [b, z, y, x], voxel coordinate for each point
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        '''
        batch_size = int(voxel_coors[:, 0].max().item()) + 1
        device = voxel_feats.device

        # TODO: Potentially add fake voxels
        gt_dict = {}

        # Points per voxel
        vx, vy, vz = self.sparse_shape
        max_num_voxels = batch_size * vx * vy * vz
        point_indices = (
            point_coors[:, 0] * vz * vy * vx +  # batch
            point_coors[:, 1] * vy * vx +  # z
            point_coors[:, 2] * vx +  # y
            point_coors[:, 3]  # x
        ).long()
        voxel_indices = (
            voxel_coors[:, 0] * vz * vy * vx +  # batch
            voxel_coors[:, 1] * vy * vx +  # z
            voxel_coors[:, 2] * vx +  # y
            voxel_coors[:, 3]  # x
        ).long()
        n_points_per_voxel_with_zeros = torch.bincount(point_indices)
        point_indices_unique = n_points_per_voxel_with_zeros.nonzero()
        n_points_per_voxel = n_points_per_voxel_with_zeros[voxel_indices]
        gt_dict["num_points_per_voxel"] = n_points_per_voxel
        assert (n_points_per_voxel > 0).all(), "Exists voxel without connected points"
        assert len(point_indices_unique) == len(voxel_indices), \
            "There is a mismatch between point indices and voxel indices"
        assert (point_indices_unique == voxel_indices.sort()).all(), \
            "There is a mismatch between point indices and voxel indices"

        # Get points per voxel
        points_rel_center = low_level_point_feature[:, -3:]
        assert self.pred_dims in [2, 3], "Either use x and y or x, y, and z"
        points_rel_center = points_rel_center[:, :self.pred_dims].clone()
        pointr_rel_norm = 1/torch.tensor(self.voxel_size, device=device).view(1, -1)
        points_rel_center = points_rel_center*pointr_rel_norm  # x,y,z all in range [-1, 1]

        shuffle = torch.argsort(torch.rand(len(point_indices)))  # Shuffle to drop random points
        restore = torch.argsort(shuffle)
        inner_voxel_inds = get_inner_win_inds(point_indices[shuffle])[restore]  # fixes one index per point per voxel
        drop_mask = inner_voxel_inds < self.drop_points_th

        points_rel_center = points_rel_center[drop_mask]
        inner_voxel_inds = inner_voxel_inds[drop_mask].long()
        dropped_point_indices = point_indices[drop_mask].long()

        gt_points = torch.zeros((max_num_voxels, self.drop_points_th, 3), device=device, dtype=points_rel_center.dtype)
        gt_points_padding = torch.zeros((max_num_voxels, self.drop_points_th), device=device, dtype=torch.long)
        gt_points[dropped_point_indices, inner_voxel_inds] = points_rel_center
        gt_points_padding[dropped_point_indices, inner_voxel_inds] = 1  # padded -> 0, not_padded -> 1
        gt_dict["points_per_voxel"] = gt_points[voxel_indices]
        gt_dict["points_per_voxel_padding"] = gt_points_padding[voxel_indices]

        test_mask = n_points_per_voxel < 100
        _n_points_per_voxel = gt_dict["points_per_voxel_padding"].sum(1)
        assert len(gt_dict["points_per_voxel"]) == len(voxel_feats), "Wrong number of point collections"
        assert (_n_points_per_voxel[test_mask] == n_points_per_voxel[test_mask]).all(), \
            "Mismatch between counted points per voxel and found points per voxel"
        assert (_n_points_per_voxel[~test_mask] == 100).all(), \
            "Error when dropping points for voxels with to many points"

        # Masking voxels: True -> masked, False -> unmasked
        mask = torch.rand(len(voxel_feats), device=device) < self.masking_ratio
        masked_idx, unmasked_idx = mask.nonzero().ravel(), (~mask).nonzero().ravel()
        n_masked_voxels, n_unmasked_voxels = len(masked_idx), len(unmasked_idx)

        # Get info for decoder input, Might drop voxels
        voxel_info_decoder = super().forward(voxel_feats, voxel_coors, batch_size=None)
        assert len(voxel_info_decoder["voxel_feats"]) == len(voxel_feats), "Dropping is not allowed for reconstruction"

        unmasked_voxels = voxel_feats[unmasked_idx]
        unmasked_voxel_coors = voxel_coors[unmasked_idx]
        # Get info for encoder input, Might drop voxels
        voxel_info_encoder = super().forward(unmasked_voxels, unmasked_voxel_coors, batch_size=None)
        assert len(voxel_info_encoder["voxel_feats"]) == n_unmasked_voxels, "Dropping is not allowed for reconstruction"

        voxel_info_decoder["mask"] = mask
        voxel_info_decoder["n_unmasked"] = n_unmasked_voxels
        voxel_info_decoder["n_masked"] = n_masked_voxels
        voxel_info_decoder["unmasked_idx"] = unmasked_idx
        voxel_info_decoder["masked_idx"] = masked_idx

        # Index mapping from decoder output to other
        dec2dec_input_idx = torch.argsort(voxel_info_decoder["original_index"])
        dec2masked_idx = dec2dec_input_idx[masked_idx]
        dec2unmasked_idx = dec2dec_input_idx[unmasked_idx]
        dec2enc_idx = dec2unmasked_idx[voxel_info_encoder["original_index"]]

        voxel_info_decoder["dec2input_idx"] = dec2dec_input_idx
        voxel_info_decoder["dec2unmasked_idx"] = dec2unmasked_idx
        voxel_info_decoder["dec2masked_idx"] = dec2masked_idx
        voxel_info_decoder["dec2enc_idx"] = dec2enc_idx

        # Debug - sanity check
        if self.debug:
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

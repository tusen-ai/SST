import os
from mmdet.models import BACKBONES
import numpy as np

import torch
import torch.nn as nn
import copy
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet3d.models.sst.sst_basic_block_v2 import BasicShiftBlockV2

from mmdet3d.ops import flat2window_v2, window2flat_v2, scatter_v2, build_mlp
from ipdb import set_trace
import torch_scatter
from .. import builder
from torch.utils.checkpoint import checkpoint


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@BACKBONES.register_module()
class StackedVFE(nn.Module):

    def __init__(
        self,
        num_blocks=5,
        in_channels=[],
        feat_channels=[],
        rel_mlp_hidden_dims=[],
        with_rel_mlp=True,
        with_distance=False,
        with_cluster_center=False,
        norm_cfg=dict(type='LN', eps=1e-3),
        mode='max',
        fusion='cat',
        pos_fusion='add',
        use_middle_cluster_feature=False,
        xyz_normalizer=[1.0, 1.0, 1.0],
        cat_voxel_feats=False,
        act='relu',
        dropout=0,
        unique_once=False,
        ):
        super().__init__()

        self.num_blocks = num_blocks
        self.use_middle_cluster_feature = use_middle_cluster_feature
        self.unique_once = unique_once
        
        block_list = []
        for i in range(num_blocks):
            return_point_feats = i != num_blocks-1
            kwargs = dict(
                type='DynamicClusterVFE',
                in_channels=in_channels[i],
                feat_channels=feat_channels[i],
                with_distance=with_distance,
                with_cluster_center=with_cluster_center,
                with_rel_mlp=with_rel_mlp,
                rel_mlp_hidden_dims=rel_mlp_hidden_dims[i],
                with_voxel_center=False,
                voxel_size=[0.1, 0.1, 0.1], # not used, placeholder
                point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4], # not used, placeholder
                norm_cfg=norm_cfg,
                mode=mode,
                fusion_layer=None,
                return_point_feats=return_point_feats,
                return_inv=False,
                rel_dist_scaler=10.0,
                fusion=fusion,
                pos_fusion=pos_fusion,
                xyz_normalizer=xyz_normalizer,
                cat_voxel_feats=cat_voxel_feats,
                act=act,
                dropout=dropout,
            )
            encoder = builder.build_voxel_encoder(kwargs)
            block_list.append(encoder)
        self.block_list = nn.ModuleList(block_list)
    
    def forward(self, points, features, coors, f_cluster=None):

        point_feat_list = []
        
        if self.unique_once:
            new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        else:
            new_coors = unq_inv = None
        
        out_feats = features

        cluster_feat_list = []
        for i, block in enumerate(self.block_list):
            in_feats = torch.cat([points, out_feats], 1)
            if i < self.num_blocks - 1:
                out_feats, out_cluster_feats = block(in_feats, coors, f_cluster, unq_inv_once=unq_inv, new_coors_once=new_coors)
                if self.use_middle_cluster_feature:
                    cluster_feat_list.append(out_cluster_feats)
            if i == self.num_blocks - 1:
                out_feats, out_cluster_feats, out_coors = block(in_feats, coors, f_cluster, return_both=True, unq_inv_once=unq_inv, new_coors_once=new_coors)
                cluster_feat_list.append(out_cluster_feats)
            
        final_cluster_feats = torch.cat(cluster_feat_list, dim=1)
        cluster_xyz, _ = scatter_v2(points[:, :3], coors, mode='avg', return_inv=False, unq_inv=unq_inv, new_coors=new_coors)

        if os.getenv('SAVE_CLUSTER'):
            if not hasattr(self, 'points_list'):
                self.points_list = []
            if not hasattr(self, 'coors_list'):
                self.coors_list = []
            self.points_list.append(points.cpu().numpy())
            self.coors_list.append(coors.cpu().numpy())
            if len(self.points_list) == 20:
                np.savez('/mnt/truenas/scratch/lve.fan/transdet3d/data/pkls/cluster_points.npz', points=self.points_list, coors=self.coors_list)

        return out_feats, final_cluster_feats, out_coors

@BACKBONES.register_module()
class OnceAggregation(nn.Module):

    def __init__(
        self,
        in_channel,
        hidden_dims,
        post_mlp_dims=None,
        reduced_pts_features_dim=None,
        norm_cfg=dict(type='LN', eps=1e-3),
        mode='max',
        xyz_normalizer=[1.0, 1.0, 1.0],
        act='relu',
        dropout=0,
        ):
        super().__init__()

        self.xyz_normalizer = xyz_normalizer
        self.mlp = build_mlp(in_channel, hidden_dims, norm_cfg, act=act)
        if post_mlp_dims is not None:
            self.post_mlp = build_mlp(hidden_dims[-1], post_mlp_dims, norm_cfg, act=act)
        if reduced_pts_features_dim is not None:
            self.out_pts_mlp = build_mlp(hidden_dims[-1], [reduced_pts_features_dim,], norm_cfg, act=act)
        self.mode = mode
    
    def forward(self, points, features, coors, f_cluster):

        xyz_scaler = torch.tensor(self.xyz_normalizer, dtype=f_cluster.dtype, device=f_cluster.device)
        f_cluster = f_cluster / xyz_scaler[None, :]
        x = torch.cat([features, f_cluster], dim=1)
        x = self.mlp(x)

        out_pts_feats = x
        if hasattr(self, 'out_pts_mlp'):
            out_pts_feats = self.out_pts_mlp(out_pts_feats)

        agg_feats, _ = scatter_v2(x, coors, mode=self.mode, return_inv=False)
        if hasattr(self, 'post_mlp'):
            agg_feats = self.post_mlp(agg_feats)

        cluster_xyz, out_coors = scatter_v2(points[:, :3], coors, mode='avg', return_inv=False)


        return out_pts_feats, agg_feats, out_coors
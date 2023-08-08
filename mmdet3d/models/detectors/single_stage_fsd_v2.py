import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result

from mmdet.models import build_detector
from ..builder import build_backbone, build_head, build_neck

from mmseg.models import SEGMENTORS
from .. import builder
from mmdet3d.ops import scatter_v2, Voxelization, furthest_point_sample, get_inner_win_inds, build_mlp
from scipy.sparse.csgraph import connected_components
from mmdet.core import multi_apply
from .single_stage import SingleStage3DDetector
from mmdet3d.models.segmentors.base import Base3DSegmentor
from mmdet3d.utils import TorchTimer

# timer = TorchTimer(100)

from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential


def filter_almost_empty(coors, min_points):
    new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    cnt_per_point = unq_cnt[unq_inv]
    valid_mask = cnt_per_point >= min_points
    return valid_mask


@DETECTORS.register_module()
class SingleStageFSDV2(SingleStage3DDetector):

    def __init__(self,
                 backbone,
                 segmentor,
                 voxel_layer=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 neck=None,
                 virtual_point_projector=None,
                 pre_voxel_encoder=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 multiscale_cfg=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)

        self.runtime_info = dict()

        if voxel_layer is not None:
            self.voxel_layer = Voxelization(**voxel_layer)

        if voxel_encoder is not None:
            self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
            self.virtual_voxel_size = voxel_encoder['voxel_size']
            self.point_cloud_range = voxel_encoder['point_cloud_range']

        if middle_encoder is not None:
            self.middle_encoder = builder.build_middle_encoder(middle_encoder)

        self.segmentor = build_detector(segmentor)
        self.use_multiscale_features = self.segmentor.use_multiscale_features
        self.head_type = bbox_head['type']
        self.num_classes = bbox_head['num_classes']

        self.cfg = self.train_cfg if self.train_cfg else self.test_cfg
        self.print_info = {}
        self.as_rpn = bbox_head.get('as_rpn', False)

        vpp = virtual_point_projector
        self.virtual_proj = build_mlp(vpp['in_channels'], vpp['hidden_dims'], vpp['norm_cfg'])
        self.ori_proj = build_mlp(vpp['ori_in_channels'], vpp['ori_hidden_dims'], vpp['norm_cfg'])
        self.zero_virtual_feature = vpp.get('zero_virtual_feature', False)
        self.only_virtual = vpp.get('only_virtual', False)

        if self.as_rpn:
            self.recover_proj = build_mlp(vpp['recover_in_channels'], vpp['recover_hidden_dims'], vpp['norm_cfg'])

        self.baseline_mode = self.cfg.get('baseline_mode', False)
        if self.baseline_mode:
            self.virtual_proj = None
            self.ori_proj = None

        self.multiscale_cfg = multiscale_cfg
        if multiscale_cfg is not None:
            ms_projs = []
            for proj in multiscale_cfg['projector_hiddens']:
                this_projector = build_mlp(proj[0], proj[1:], multiscale_cfg['norm_cfg'])
                ms_projs.append(this_projector)
            self.ms_projectors = torch.nn.ModuleList(ms_projs)

    @torch.no_grad()
    @force_fp32()
    def voxelize_with_batch_idx(self, points, batch_idx):
        """Apply dynamic voxelization to points.
        """
        points = points[:, :3]
        device = points.device
        voxel_size = torch.tensor(self.virtual_voxel_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)

        res_coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        res_coors = res_coors[:, [2, 1, 0]] # to zyx order

        coors_batch = torch.cat([batch_idx[:, None], res_coors], dim=1)

        return coors_batch

    def clip_points(self, points, pc_range):
        eps = 1e-5
        points[:, 0] = points[:, 0].clamp(min=pc_range[0] + eps, max=pc_range[3] - eps)
        points[:, 1] = points[:, 1].clamp(min=pc_range[1] + eps, max=pc_range[4] - eps)
        points[:, 2] = points[:, 2].clamp(min=pc_range[2] + eps, max=pc_range[5] - eps)
        return points

    def recover_point_features(self, out_voxel_feats, out_coors, out_sparse_shape, cat_pts, cat_batch_idx, voxel_encoder_coors, voxel_encoder_inv):

        is_same = (out_coors == voxel_encoder_coors).all()
        device = out_voxel_feats.device

        if is_same:
            # Meaning that spconv does not change the spatial layout of voxels
            voxel_size = torch.tensor(self.virtual_voxel_size, device=device) # correct only if is_same 
            pc_range = torch.tensor(self.point_cloud_range, device=device)

            voxel_coors_per_pts = out_coors[voxel_encoder_inv]
            feat_per_pts = out_voxel_feats[voxel_encoder_inv]
            voxel_center_per_pts = (voxel_coors_per_pts[:, [3, 2, 1]] + 0.5) * voxel_size[None, :] + pc_range[None, :3]
            offset_per_pts = voxel_center_per_pts - cat_pts
            assert (offset_per_pts.abs() < voxel_size[None, :] / 2 + 1e5).all()

            offset_per_pts = offset_per_pts / voxel_size[None, :] * 2 # normalize

            proj_input = torch.cat([feat_per_pts, offset_per_pts], 1)
            out = self.recover_proj(proj_input)

            return out

        else:
            raise NotImplementedError



    def extract_feat(self, sampled_dict, origin_dict, gt_bboxes_3d=None, multiscale_features=None):
        """Extract features from points."""
        if self.baseline_mode:
            return self.extract_feat_baseline(sampled_dict, origin_dict, gt_bboxes_3d, multiscale_features)

        sampled_pts = sampled_dict['seg_points']
        sampled_centers = sampled_dict['center_preds']
        sampled_logits = sampled_dict['seg_logits']
        sampled_feats = sampled_dict['seg_feats']
        sampled_batch_idx = sampled_dict['batch_idx']
        device = sampled_pts.device

        # the predicted centers might be out-of-range
        sampled_centers = self.clip_points(sampled_centers, self.point_cloud_range)
        # sampled_centers

        offset = (sampled_centers - sampled_pts[:, :3]) / 10 # hardcode a normalizer
        proj_input = torch.cat([sampled_feats, offset, sampled_logits, sampled_pts[:, 3:]], 1)
        vir_pts_feat = self.virtual_proj(proj_input)

        if self.zero_virtual_feature:
            vir_pts_feat = vir_pts_feat * 0

        ori_pts = origin_dict['seg_points']
        ori_batch_idx = origin_dict['batch_idx']
        ori_pts_feat = origin_dict['seg_feats']
        ori_pts_feat = self.ori_proj(ori_pts_feat)

        cat_pts = torch.cat([ori_pts[:, :3], sampled_centers], 0)
        cat_feat = torch.cat([ori_pts_feat, vir_pts_feat], 0)
        cat_batch_idx = torch.cat([ori_batch_idx, sampled_batch_idx], 0)

        coors = self.voxelize_with_batch_idx(cat_pts, cat_batch_idx)

        voxel_encoder_input = torch.cat([cat_pts, cat_feat], 1)
        voxel_feats, voxel_coors, unq_inv = self.voxel_encoder(voxel_encoder_input, coors, return_inv=True)

        pts_indicators = torch.cat(
            [
                torch.zeros(len(ori_pts), device=device, dtype=torch.float),
                torch.ones( len(sampled_centers), device=device, dtype=torch.float)
            ]
        )
        voxel_indicators, scatter_coors = scatter_v2(pts_indicators, coors, mode='avg', return_inv=False)
        assert (scatter_coors == voxel_coors).all()
        virtual_mask = voxel_indicators > 0

        batch_size = voxel_coors[:, 0].max().item() + 1

        if multiscale_features is not None:
            voxel_feats, voxel_coors, singlescale_mask = self.multiscale_fusion(multiscale_features, voxel_feats, voxel_coors)

        if self.only_virtual:
            assert multiscale_features is None
            voxel_feats = voxel_feats[virtual_mask]
            voxel_coors = voxel_coors[virtual_mask]

        out_voxel_feats, out_coors, sparse_shape = self.backbone(voxel_feats, voxel_coors, batch_size)

        if multiscale_features is not None:
            out_voxel_feats = out_voxel_feats[singlescale_mask]
            out_coors = out_coors[singlescale_mask]
            voxel_coors = voxel_coors[singlescale_mask] # in fact, out_coors and voxel_coors are same

        # get voxel center xyz
        voxel_size = torch.tensor(self.virtual_voxel_size, device=device) # correct only if is_same 
        pc_range = torch.tensor(self.point_cloud_range, device=device)
        voxel_centers = (out_coors[:, [3, 2, 1]] + 0.5) * voxel_size[None, :] + pc_range[None, :3]

        if self.only_virtual:
            virtual_voxel_feats = out_voxel_feats
            virtual_coors = out_coors
            virtual_centers = voxel_centers
        else:
            virtual_voxel_feats = out_voxel_feats[virtual_mask]
            virtual_coors = out_coors[virtual_mask]
            virtual_centers = voxel_centers[virtual_mask]

        out_dict = dict(
            virtual_feats=virtual_voxel_feats,
            virtual_coors=virtual_coors,
            virtual_centers=virtual_centers,
            # voxel_indicators=voxel_indicators
            sparse_shape=sparse_shape,
        )

        if self.training:
            self.print_info['num_virtual'] = virtual_voxel_feats.new_ones(1) * len(virtual_voxel_feats)

            alpha = self.train_cfg.get('centroid_alpha', None)
            if alpha is not None:
                gt_fg_mask = self.get_batched_gt_fg_mask(cat_pts[:, :3], cat_batch_idx, gt_bboxes_3d)
                alpha_mask = (~gt_fg_mask).float() * alpha + gt_fg_mask.float()
                sum_centroid, _ = scatter_v2(alpha_mask[:, None] * cat_pts[:, :3], coors, mode='sum', return_inv=False)
                sum_alpha, _ = scatter_v2(alpha_mask[:, None], coors, mode='sum', return_inv=False)
                assert (sum_alpha >= alpha).all()
                voxel_centroid = sum_centroid / sum_alpha
            else:
                voxel_centroid, _ = scatter_v2(cat_pts[:, :3], coors, mode='avg', return_inv=False)

            voxel_centroid = voxel_centroid[virtual_mask]
            out_dict['virtual_centroid'] = voxel_centroid
            assert ((voxel_centroid - virtual_centers).abs() < voxel_size / 2 + 1e-3).all()

        if self.as_rpn:
            # need pts information for GroupCorrection
            out_pts_feats = self.recover_point_features(out_voxel_feats, out_coors, sparse_shape, cat_pts, cat_batch_idx, voxel_coors, unq_inv)
            out_dict['pts_feats'] = out_pts_feats
            out_dict['pts_xyz'] = cat_pts
            out_dict['pts_indicators'] = pts_indicators
            out_dict['pts_batch_inds'] = cat_batch_idx

        return out_dict

    def extract_feat_baseline(self, sampled_dict, origin_dict, gt_bboxes_3d=None, multiscale_features=None):
        """Extract features from points."""
        sampled_pts = sampled_dict['seg_points']
        # sampled_centers = sampled_dict['center_preds']
        sampled_centers = sampled_pts[:, :3]
        sampled_logits = sampled_dict['seg_logits']
        sampled_feats = sampled_dict['seg_feats']
        sampled_batch_idx = sampled_dict['batch_idx']
        device = sampled_pts.device

        # the predicted centers might be out-of-range
        # sampled_centers = self.clip_points(sampled_centers, self.point_cloud_range)
        # sampled_centers

        # offset = (sampled_centers - sampled_pts[:, :3]) / 10 # hardcode a normalizer
        # proj_input = torch.cat([sampled_feats, offset, sampled_logits, sampled_pts[:, 3:]], 1)
        # vir_pts_feat = self.virtual_proj(proj_input)

        vir_pts_feat = sampled_feats

        ori_pts = origin_dict['seg_points']
        ori_batch_idx = origin_dict['batch_idx']
        ori_pts_feat = origin_dict['seg_feats']
        # ori_pts_feat = self.ori_proj(ori_pts_feat)

        cat_pts = torch.cat([ori_pts[:, :3], sampled_centers], 0)
        cat_feat = torch.cat([ori_pts_feat, vir_pts_feat], 0)
        cat_batch_idx = torch.cat([ori_batch_idx, sampled_batch_idx], 0)

        coors = self.voxelize_with_batch_idx(cat_pts, cat_batch_idx)

        voxel_encoder_input = torch.cat([cat_pts, cat_feat], 1)
        voxel_feats, voxel_coors, unq_inv = self.voxel_encoder(voxel_encoder_input, coors, return_inv=True)

        pts_indicators = torch.cat(
            [
                torch.zeros(len(ori_pts), device=device, dtype=torch.float),
                torch.ones( len(sampled_centers), device=device, dtype=torch.float)
            ]
        )
        voxel_indicators, scatter_coors = scatter_v2(pts_indicators, coors, mode='avg', return_inv=False)
        assert (scatter_coors == voxel_coors).all()

        batch_size = voxel_coors[:, 0].max().item() + 1
        # assert batch_size == 2, 'Develop assertion, ok to delete'

        if multiscale_features is not None:
            voxel_feats, voxel_coors, singlescale_mask = self.multiscale_fusion(multiscale_features, voxel_feats, voxel_coors)

        out_voxel_feats, out_coors, sparse_shape = self.backbone(voxel_feats, voxel_coors, batch_size)

        if multiscale_features is not None:
            out_voxel_feats = out_voxel_feats[singlescale_mask]
            out_coors = out_coors[singlescale_mask]
            voxel_coors = voxel_coors[singlescale_mask] # in fact, out_coors and voxel_coors are same

        # get voxel center xyz
        voxel_size = torch.tensor(self.virtual_voxel_size, device=device) # correct only if is_same 
        pc_range = torch.tensor(self.point_cloud_range, device=device)
        voxel_centers = (out_coors[:, [3, 2, 1]] + 0.5) * voxel_size[None, :] + pc_range[None, :3]

        virtual_mask = voxel_indicators > 0
        virtual_voxel_feats = out_voxel_feats[virtual_mask]
        virtual_coors = out_coors[virtual_mask]
        virtual_centers = voxel_centers[virtual_mask]

        out_dict = dict(
            virtual_feats=virtual_voxel_feats,
            virtual_coors=virtual_coors,
            virtual_centers=virtual_centers,
            # voxel_indicators=voxel_indicators
            sparse_shape=sparse_shape,
        )

        if self.training:
            self.print_info['num_virtual'] = virtual_voxel_feats.new_ones(1) * len(virtual_voxel_feats)

            alpha = self.train_cfg.get('centroid_alpha', None)
            if alpha is not None:
                gt_fg_mask = self.get_batched_gt_fg_mask(cat_pts[:, :3], cat_batch_idx, gt_bboxes_3d)
                alpha_mask = (~gt_fg_mask).float() * alpha + gt_fg_mask.float()
                sum_centroid, _ = scatter_v2(alpha_mask[:, None] * cat_pts[:, :3], coors, mode='sum', return_inv=False)
                sum_alpha, _ = scatter_v2(alpha_mask[:, None], coors, mode='sum', return_inv=False)
                assert (sum_alpha >= alpha).all()
                voxel_centroid = sum_centroid / sum_alpha
            else:
                voxel_centroid, _ = scatter_v2(cat_pts[:, :3], coors, mode='avg', return_inv=False)

            voxel_centroid = voxel_centroid[virtual_mask]
            out_dict['virtual_centroid'] = voxel_centroid
            assert ((voxel_centroid - virtual_centers).abs() < voxel_size / 2 + 1e-3).all()

        if self.as_rpn:
            # need pts information for GroupCorrection
            out_pts_feats = self.recover_point_features(out_voxel_feats, out_coors, sparse_shape, cat_pts, cat_batch_idx, voxel_coors, unq_inv)
            out_dict['pts_feats'] = out_pts_feats
            out_dict['pts_xyz'] = cat_pts
            out_dict['pts_indicators'] = pts_indicators
            out_dict['pts_batch_inds'] = cat_batch_idx

        return out_dict

    def multiscale_fusion(self, ms_data, voxel_feats, coors):

        cfg = self.multiscale_cfg

        ms_data = [ms_data[l] for l in cfg['multiscale_levels']]
        ms_feats = [ self.ms_projectors[i](ms_data[i].features) for i in range(len(ms_data)) ]
        ms_coors = [ self.ms_coors_proj(data.indices, data.spatial_shape) for data in ms_data]

        num_add_feats = sum([len(f) for f in ms_feats])

        cat_feats = torch.cat([voxel_feats,] + ms_feats, 0)
        cat_coors = torch.cat([coors,] + ms_coors, 0)
        indicators = torch.cat([voxel_feats.new_ones(len(voxel_feats), 1), voxel_feats.new_zeros(num_add_feats, 1)], 0)

        out_feats, out_coors = scatter_v2(cat_feats, cat_coors, mode=cfg['fusion_mode'], return_inv=False)
        out_indicators, _ = scatter_v2(indicators, cat_coors, mode='max', return_inv=False)
        out_indicators = out_indicators.squeeze()

        singlescale_mask = out_indicators == 1
        assert singlescale_mask.sum() == len(voxel_feats)

        return out_feats, out_coors, singlescale_mask

    def ms_coors_proj(self, coors, sparse_shape):

        # support float coors, need inference test
        # cfg = self.multiscale_cfg
        # tgt_sp = cfg['target_sparse_shape']
        # bev_stride = tgt_sp[1] / sparse_shape[1]
        # assert bev_stride == tgt_sp[2] / sparse_shape[2]
        # z_stride = tgt_sp[0] / sparse_shape[0]

        # out_coors = coors.clone()
        # out_coors[:, 1] = (coors[:, 1] * z_stride + z_stride / 2).int()
        # out_coors[:, 2] = (coors[:, 2] * bev_stride + bev_stride / 2).int()
        # out_coors[:, 3] = (coors[:, 3] * bev_stride + bev_stride / 2).int()

        # assert out_coors[:, 1].max().item() < tgt_sp[0]
        # assert out_coors[:, 2].max().item() < tgt_sp[1]
        # assert out_coors[:, 3].max().item() < tgt_sp[2]

        cfg = self.multiscale_cfg
        tgt_sp = cfg['target_sparse_shape']
        bev_stride = tgt_sp[1] // sparse_shape[1]
        assert bev_stride == tgt_sp[2] / sparse_shape[2]
        z_stride = tgt_sp[0] // sparse_shape[0]

        assert z_stride >= 1
        assert bev_stride >= 1

        out_coors = coors.clone()
        out_coors[:, 1] = coors[:, 1] * z_stride + z_stride // 2
        out_coors[:, 2] = coors[:, 2] * bev_stride + bev_stride // 2
        out_coors[:, 3] = coors[:, 3] * bev_stride + bev_stride // 2

        assert out_coors[:, 1].max().item() < tgt_sp[0]
        assert out_coors[:, 2].max().item() < tgt_sp[1]
        assert out_coors[:, 3].max().item() < tgt_sp[2]

        return out_coors



    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None,
                      runtime_info=None):
        if runtime_info is not None:
            self.runtime_info = runtime_info # stupid way to get arguements from children class
        losses = {}
        gt_bboxes_3d = [b[l>=0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d = [l[l>=0] for l in gt_labels_3d]

        bsz = len(points)

        seg_out_dict = self.segmentor(points=points, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, as_subsegmentor=True)

        seg_feats = seg_out_dict['seg_feats']
        if self.train_cfg.get('detach_segmentor', False):
            seg_feats = seg_feats.detach()
        seg_loss = seg_out_dict['losses']
        losses.update(seg_loss)

        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],
            seg_logits=seg_out_dict['seg_logits'].detach(),
            seg_vote_preds=seg_out_dict['seg_vote_preds'].detach(),
            seg_feats=seg_feats,
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets=seg_out_dict['offsets'].detach(),
        )

        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'], gt_bboxes_3d, gt_labels_3d) # per cls list in sampled_out

        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds', 'batch_idx'])

        extract_output = self.extract_feat(combined_out, dict_to_sample, gt_bboxes_3d=gt_bboxes_3d, multiscale_features=seg_out_dict['decoder_features'])

        voxel_feats = extract_output['virtual_feats']
        voxel_coors = extract_output['virtual_coors']
        voxel_xyz = extract_output['virtual_centers']

        outs = self.bbox_head(voxel_feats)

        loss_inputs = (outs['cls_logits'], outs['reg_preds']) + (voxel_xyz, voxel_coors[:, 0]) + (gt_bboxes_3d, gt_labels_3d, img_metas)
        det_loss = self.bbox_head.loss(
            *loss_inputs, iou_logits=outs.get('iou_logits', None), gt_bboxes_ignore=gt_bboxes_ignore, aux_xyz=extract_output['virtual_centroid'])

        if hasattr(self.bbox_head, 'print_info'):
            self.print_info.update(self.bbox_head.print_info)
        losses.update(det_loss)
        losses.update(self.print_info)

        if self.as_rpn:
            output_dict = dict(
                rpn_losses=losses,
                cls_logits=outs['cls_logits'],
                reg_preds=outs['reg_preds'],
                voxel_xyz=voxel_xyz,
                voxel_batch_inds=voxel_coors[:, 0],
                pts_feats=extract_output['pts_feats'],
                pts_xyz=extract_output['pts_xyz'],
                pts_indicators=extract_output['pts_indicators'],
                pts_batch_inds=extract_output['pts_batch_inds'],
            )
            return output_dict
        else:
            return losses


    def combine_classes(self, data_dict, name_list):
        out_dict = {}
        for name in data_dict:
            if name in name_list:
                out_dict[name] = torch.cat(data_dict[name], 0)
        return out_dict

    def pre_voxelize(self, data_dict):
        raise NotImplementedError('No need to use prevoxelization anymore in FSDV2')
        batch_idx = data_dict['batch_idx']
        points = data_dict['seg_points']

        voxel_size = torch.tensor(self.cfg.pre_voxelization_size, device=batch_idx.device)
        pc_range = torch.tensor(self.cluster_assigner.point_cloud_range, device=points.device)
        coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)

        new_coors, unq_inv  = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)

        voxelized_data_dict = {}
        for data_name in data_dict:
            data = data_dict[data_name]
            if data.dtype in (torch.float, torch.float16):
                voxelized_data, voxel_coors = scatter_v2(data, coors, mode='avg', return_inv=False, new_coors=new_coors, unq_inv=unq_inv)
                voxelized_data_dict[data_name] = voxelized_data

        voxelized_data_dict['batch_idx'] = voxel_coors[:, 0]
        return voxelized_data_dict


    def simple_test(self, points, img_metas, imgs=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentaiton."""
        if gt_bboxes_3d is not None:
            gt_bboxes_3d = gt_bboxes_3d[0]
            gt_labels_3d = gt_labels_3d[0]
            assert isinstance(gt_bboxes_3d, list)
            assert isinstance(gt_labels_3d, list)
            assert len(gt_bboxes_3d) == len(gt_labels_3d) == 1, 'assuming single sample testing'

        bsz = len(points)

        seg_out_dict = self.segmentor.simple_test(points, img_metas, rescale=False)

        seg_feats = seg_out_dict['seg_feats']

        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],
            seg_logits=seg_out_dict['seg_logits'],
            seg_vote_preds=seg_out_dict['seg_vote_preds'],
            seg_feats=seg_feats,
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets = seg_out_dict['offsets']
        )
        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'], gt_bboxes_3d, gt_labels_3d) # per cls list in sampled_out

        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds', 'batch_idx'])

        extract_output = self.extract_feat(combined_out, dict_to_sample, multiscale_features=seg_out_dict['decoder_features'])

        voxel_feats = extract_output['virtual_feats']
        voxel_coors = extract_output['virtual_coors']
        voxel_xyz = extract_output['virtual_centers']

        outs = self.bbox_head(voxel_feats)

        bbox_list = self.bbox_head.get_bboxes(
            outs['cls_logits'], outs['reg_preds'],
            voxel_xyz, voxel_coors[:, 0], img_metas,
            rescale=rescale,
            iou_logits=outs.get('iou_logits', None))

        if self.as_rpn:
            output_dict = dict(
                pts_feats=extract_output['pts_feats'],
                pts_xyz=extract_output['pts_xyz'],
                pts_indicators=extract_output['pts_indicators'],
                pts_batch_inds=extract_output['pts_batch_inds'],
                proposal_list=bbox_list
            )
            return output_dict
        else:
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        return NotImplementedError


    def sample(self, dict_to_sample, offset, gt_bboxes_3d=None, gt_labels_3d=None):

        if self.cfg.get('group_sample', False):
            return self.group_sample(dict_to_sample, offset)

        if self.cfg.get('batched_group_sample', False):
            return self.batched_group_sample(dict_to_sample, offset)

        cfg = self.train_cfg if self.training else self.test_cfg

        seg_logits = dict_to_sample['seg_logits']
        assert (seg_logits < 0).any() # make sure no sigmoid applied

        if seg_logits.size(1) == self.num_classes:
            seg_scores = seg_logits.sigmoid()
        else:
            raise NotImplementedError

        offset = offset.reshape(-1, self.num_classes, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls

        batch_idx = dict_to_sample['batch_idx']
        batch_size = batch_idx.max().item() + 1
        for cls in range(self.num_classes):
            cls_score_thr = cfg['score_thresh'][cls]

            fg_mask = self.get_fg_mask(seg_scores, seg_points, cls, batch_idx, gt_bboxes_3d, gt_labels_3d)

            if len(torch.unique(batch_idx[fg_mask])) < batch_size:
                one_random_pos_per_sample = self.get_sample_beg_position(batch_idx, fg_mask)
                fg_mask[one_random_pos_per_sample] = True # at least one point per sample

            fg_mask_list.append(fg_mask)

            this_offset = offset[fg_mask, cls, :]
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)


        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict

    def get_sample_beg_position(self, batch_idx, fg_mask):
        assert batch_idx.shape == fg_mask.shape
        inner_inds = get_inner_win_inds(batch_idx.contiguous())
        pos = torch.where(inner_inds == 0)[0]
        return pos

    def get_fg_mask(self, seg_scores, seg_points, cls_id, batch_inds, gt_bboxes_3d, gt_labels_3d):
        if self.training and self.train_cfg.get('disable_pretrain', False) and not self.runtime_info.get('enable_detection', False):
            seg_scores = seg_scores[:, cls_id]
            topks = self.train_cfg.get('disable_pretrain_topks', [100, 100, 100])
            k = min(topks[cls_id], len(seg_scores))
            top_inds = torch.topk(seg_scores, k)[1]
            fg_mask = torch.zeros_like(seg_scores, dtype=torch.bool)
            fg_mask[top_inds] = True
        else:
            seg_scores = seg_scores[:, cls_id]
            cls_score_thr = self.cfg['score_thresh'][cls_id]
            if self.training and self.runtime_info is not None:
                buffer_thr = self.runtime_info.get('threshold_buffer', 0)
            else:
                buffer_thr = 0
            fg_mask = seg_scores > cls_score_thr + buffer_thr

        # add fg points
        cfg = self.train_cfg if self.training else self.test_cfg

        if cfg.get('add_gt_fg_points', False) and self.training:
            if self.runtime_info.get('stop_add_gt_fg_points', False):
                return fg_mask
            bsz = len(gt_bboxes_3d)
            assert len(seg_scores) == len(seg_points) == len(batch_inds)
            point_list = self.split_by_batch(seg_points, batch_inds, bsz)
            gt_fg_mask_list = []

            for i, points in enumerate(point_list):

                gt_mask = gt_labels_3d[i] == cls_id
                gts = gt_bboxes_3d[i][gt_mask]

                if not gt_mask.any() or len(points) == 0:
                    gt_fg_mask_list.append(gt_mask.new_zeros(len(points), dtype=torch.bool))
                    continue

                gt_fg_mask_list.append(gts.points_in_boxes(points) > -1)

            gt_fg_mask = self.combine_by_batch(gt_fg_mask_list, batch_inds, bsz)
            fg_mask = fg_mask | gt_fg_mask


        return fg_mask

    def split_by_batch(self, data, batch_idx, batch_size):
        assert batch_idx.max().item() + 1 <= batch_size
        data_list = []
        for i in range(batch_size):
            sample_mask = batch_idx == i
            data_list.append(data[sample_mask])
        return data_list

    def combine_by_batch(self, data_list, batch_idx, batch_size):
        assert len(data_list) == batch_size
        if data_list[0] is None:
            return None
        data_shape = (len(batch_idx),) + data_list[0].shape[1:]
        full_data = data_list[0].new_zeros(data_shape)
        for i, data in enumerate(data_list):
            sample_mask = batch_idx == i
            full_data[sample_mask] = data
        return full_data

    def group_sample(self, dict_to_sample, offset):
        bsz = dict_to_sample['batch_idx'].max().item() + 1
        assert bsz == 1, "Maybe some codes need to be modified if bsz > 1, this will be updated very soon"
        # combine all classes as fg class.
        cfg = self.train_cfg if self.training else self.test_cfg

        seg_logits = dict_to_sample['seg_logits']
        assert (seg_logits < 0).any() # make sure no sigmoid applied

        assert seg_logits.size(1) == self.num_classes + 1 # we have background class
        seg_scores = seg_logits.softmax(1)

        offset = offset.reshape(-1, self.num_classes + 1, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls


        cls_score_thrs = cfg['score_thresh']
        group_names = cfg['group_names']
        class_names = cfg['class_names']
        num_groups = len(group_names)
        assert num_groups == len(cls_score_thrs)
        assert isinstance(cls_score_thrs, (list, tuple))
        grouped_score = self.gather_group_by_names(seg_scores[:, :-1]) # without background score

        for i in range(num_groups):

            fg_mask = self.get_fg_mask(grouped_score, None, i, None, None, None)

            if not fg_mask.any():
                fg_mask[0] = True # at least one point

            fg_mask_list.append(fg_mask)

            tmp_idx = []
            for name in group_names[i]:
                tmp_idx.append(class_names.index(name))

            this_offset = offset[:, tmp_idx, :] 
            this_offset = this_offset[fg_mask, ...]
            this_logits = seg_logits[:, tmp_idx]
            this_logits = this_logits[fg_mask, :]

            offset_weight = self.get_offset_weight(this_logits)
            assert torch.isclose(offset_weight.sum(1), offset_weight.new_ones(len(offset_weight))).all()
            this_offset = (this_offset * offset_weight[:, :, None]).sum(dim=1)
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)

        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict

    def batched_group_sample(self, dict_to_sample, offset):
        batch_idx = dict_to_sample['batch_idx']
        batch_size = batch_idx.max().item() + 1
        # combine all classes as fg class.
        cfg = self.train_cfg if self.training else self.test_cfg

        seg_logits = dict_to_sample['seg_logits']

        assert seg_logits.size(1) == self.num_classes + 1 # we have background class
        seg_scores = seg_logits.softmax(1)

        offset = offset.reshape(-1, self.num_classes + 1, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls


        cls_score_thrs = cfg['score_thresh']
        group_names = cfg['group_names']
        class_names = cfg['class_names']
        offset_scale = cfg.get('offset_scale', 1)
        num_groups = len(group_names)
        assert num_groups == len(cls_score_thrs)
        assert isinstance(cls_score_thrs, (list, tuple))
        grouped_score = self.gather_group_by_names(seg_scores[:, :-1]) # without background score

        for i in range(num_groups):

            fg_mask = self.get_fg_mask(grouped_score, None, i, None, None, None)

            if len(torch.unique(batch_idx[fg_mask])) < batch_size:
                one_random_pos_per_sample = self.get_sample_beg_position(batch_idx, fg_mask)
                fg_mask[one_random_pos_per_sample] = True # at least one point per sample

            fg_mask_list.append(fg_mask)

            tmp_idx = []
            for name in group_names[i]:
                tmp_idx.append(class_names.index(name))

            this_offset = offset[:, tmp_idx, :] 
            this_offset = this_offset[fg_mask, ...]
            this_logits = seg_logits[:, tmp_idx]
            this_logits = this_logits[fg_mask, :]

            offset_weight = self.get_offset_weight(this_logits)
            assert torch.isclose(offset_weight.sum(1), offset_weight.new_ones(len(offset_weight))).all()
            this_offset = (this_offset * offset_weight[:, :, None]).sum(dim=1)
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + offset_scale * this_offset
            center_preds_list.append(this_centers)

        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict

    def get_offset_weight(self, seg_logit):
        mode = self.cfg['offset_weight']
        if mode == 'max':
            weight = ((seg_logit - seg_logit.max(1)[0][:, None]).abs() < 1e-6).float()
            assert ((weight == 1).any(1)).all()
            weight = weight / weight.sum(1)[:, None] # in case of two max values
            return weight
        else:
            raise NotImplementedError

    def gather_group(self, scores, group_lens):
        assert (scores >= 0).all()
        score_per_group = []
        beg = 0
        for group_len in group_lens:
            end = beg + group_len
            score_this_g = scores[:, beg:end].sum(1)
            score_per_group.append(score_this_g)
            beg = end
        assert end == scores.size(1) == sum(group_lens)
        gathered_score = torch.stack(score_per_group, dim=1)
        assert gathered_score.size(1) == len(group_lens)
        return  gathered_score

    def gather_group_by_names(self, scores):
        groups = self.cfg['group_names']
        class_names = self.cfg['class_names']
        assert (scores >= 0).all()
        score_per_group = []
        for g in groups:
            tmp_idx = []
            for name in g:
                tmp_idx.append(class_names.index(name))
            score_per_group.append(scores[:, tmp_idx].sum(1))

        gathered_score = torch.stack(score_per_group, dim=1)
        return  gathered_score

    def get_batched_gt_fg_mask(self, points, batch_inds, gt_bboxes_3d):
        bsz = batch_inds.max().item() + 1
        point_list = self.split_by_batch(points, batch_inds, bsz)
        gt_fg_mask_list = []
        assert len(point_list) == len(gt_bboxes_3d)

        for i, points in enumerate(point_list):

            gts = gt_bboxes_3d[i]

            if len(gts) == 0 or len(points) == 0:
                gt_fg_mask_list.append(points.new_zeros(len(points), dtype=torch.bool))
                continue

            gt_fg_mask_list.append(gts.points_in_boxes(points) > -1)

        gt_fg_mask = self.combine_by_batch(gt_fg_mask_list, batch_inds, bsz)
        return gt_fg_mask
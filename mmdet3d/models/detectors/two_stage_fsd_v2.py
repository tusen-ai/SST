from .single_stage_fsd_v2 import SingleStageFSDV2
import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.ops import scatter_v2
from .. import builder

from ipdb import set_trace

@DETECTORS.register_module()
class FSDV2(SingleStageFSDV2):

    def __init__(self,
                 backbone,
                 segmentor,
                 voxel_layer=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 neck=None,
                 virtual_point_projector=None,
                 bbox_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 multiscale_cfg=None,):
        super().__init__(
            backbone=backbone,
            segmentor=segmentor,
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            neck=neck,
            virtual_point_projector=virtual_point_projector,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            multiscale_cfg=multiscale_cfg,
        )

        # update train and test cfg here for now
        rcnn_train_cfg = train_cfg.rcnn if train_cfg else None
        roi_head.update(train_cfg=rcnn_train_cfg)
        roi_head.update(test_cfg=test_cfg.rcnn)
        roi_head.pretrained = pretrained

        if 'with_virtual' in roi_head.roi_extractor:
            self.with_virtual = roi_head.roi_extractor.pop('with_virtual')
        else:
            self.with_virtual = False

        self.roi_head = builder.build_head(roi_head)
        self.num_classes = self.bbox_head.num_classes
        self.runtime_info = dict()

        self.pc_range = voxel_encoder['point_cloud_range']


    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        gt_bboxes_3d = [b[l>=0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d = [l[l>=0] for l in gt_labels_3d]


        losses = {}
        rpn_outs = super().forward_train(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_ignore=gt_bboxes_ignore,
            runtime_info=self.runtime_info
        )
        losses.update(rpn_outs['rpn_losses'])

        proposal_list = self.bbox_head.get_bboxes(
            rpn_outs['cls_logits'], rpn_outs['reg_preds'], rpn_outs['voxel_xyz'], rpn_outs['voxel_batch_inds'], img_metas
        )

        assert len(proposal_list) == len(gt_bboxes_3d)


        pts_xyz = rpn_outs['pts_xyz']
        pts_feats = rpn_outs['pts_feats']
        pts_batch_inds = rpn_outs['pts_batch_inds']
        pts_indicators = rpn_outs['pts_indicators']

        if not self.with_virtual:
            ori_mask = pts_indicators == 0
            pts_xyz = pts_xyz[ori_mask]
            pts_feats = pts_feats[ori_mask]
            pts_batch_inds = pts_batch_inds[ori_mask]

        pts_xyz, pts_feats, pts_batch_inds = self.pre_voxelize(pts_xyz, pts_feats, pts_batch_inds)

        # sort for the needs of ROI extractor
        pts_batch_inds, inds = pts_batch_inds.sort()
        pts_xyz = pts_xyz[inds]
        pts_feats = pts_feats[inds]

        roi_losses = self.roi_head.forward_train(
            pts_xyz,
            pts_feats,
            pts_batch_inds,
            img_metas,
            proposal_list,
            gt_bboxes_3d,
            gt_labels_3d,
        )

        losses.update(roi_losses)

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):


        rpn_outs = super().simple_test(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
        )

        proposal_list = rpn_outs['proposal_list']

        if self.test_cfg.get('skip_rcnn', False):
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in proposal_list
            ]
            return bbox_results

        pts_xyz = rpn_outs['pts_xyz']
        pts_feats = rpn_outs['pts_feats']
        pts_batch_inds = rpn_outs['pts_batch_inds']
        pts_indicators = rpn_outs['pts_indicators']

        if not self.with_virtual:
            ori_mask = pts_indicators == 0
            pts_xyz = pts_xyz[ori_mask]
            pts_feats = pts_feats[ori_mask]
            pts_batch_inds = pts_batch_inds[ori_mask]

        pts_xyz, pts_feats, pts_batch_inds = self.pre_voxelize(pts_xyz, pts_feats, pts_batch_inds)

        # sort for the needs of ROI extractor
        pts_batch_inds, inds = pts_batch_inds.sort()
        pts_xyz = pts_xyz[inds]
        pts_feats = pts_feats[inds]

        results = self.roi_head.simple_test(
            pts_xyz,
            pts_feats,
            pts_batch_inds,
            img_metas,
            proposal_list,
            gt_bboxes_3d,
            gt_labels_3d,
        )

        return results


    def pre_voxelize(self, points, features, batch_idx):
        """Apply dynamic voxelization to points.
        """
        if self.training:
            voxel_size = self.train_cfg.get('pre_2nd_voxelization', None)
        else:
            voxel_size = self.test_cfg.get('pre_2nd_voxelization', None)

        if voxel_size is None:
            return points, features, batch_idx

        points_xyz = points[:, :3]
        device = points.device

        voxel_size = torch.tensor(voxel_size, device=device)
        pc_range = torch.tensor(self.pc_range, device=device)

        res_coors = torch.div(points_xyz - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        res_coors = res_coors[:, [2, 1, 0]] # to zyx order

        coors_batch = torch.cat([batch_idx[:, None], res_coors], dim=1)

        new_coors, unq_inv = torch.unique(coors_batch, return_inverse=True, return_counts=False, dim=0)

        new_points, _ = scatter_v2(points, coors_batch, mode='avg', return_inv=False, new_coors=new_coors, unq_inv=unq_inv)
        new_feats, _ = scatter_v2(features, coors_batch, mode='avg', return_inv=False, new_coors=new_coors, unq_inv=unq_inv)

        return new_points, new_feats, new_coors[:, 0]
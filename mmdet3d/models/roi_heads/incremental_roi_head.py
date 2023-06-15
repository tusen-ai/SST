import warnings
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mmdet3d.core import AssignResult
from mmdet3d.core.bbox import bbox3d2result, bbox3d2roi, LiDARInstance3DBoxes
from mmdet.core import build_assigner, build_sampler
from mmdet.models import HEADS
from ..builder import build_head, build_roi_extractor
from .fsd_roi_head import GroupCorrectionHead
# from .traj_utils.traj_generator import TrajGenerator
# from .traj_utils.traj_feature_extractor import TrajFeatureExtractor, TrajFeatureAggregator

from ipdb import set_trace
from mmdet3d.utils import TorchTimer
timer = TorchTimer(-1)


@HEADS.register_module()
class IncrementalROIHead(GroupCorrectionHead):

    def __init__(self,
                 num_classes=3,
                 roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 incremental_cfg=None,
                 traj_feature_extractor=None,
                 traj_feature_aggregator=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            num_classes=num_classes,
            roi_extractor=roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg
        )

        self.traj_checkpointing = traj_feature_extractor.pop('checkpointing', False)
        self.traj_feature_extractor = TrajFeatureExtractor(**traj_feature_extractor)
        self.traj_feature_aggregator = TrajFeatureAggregator(**traj_feature_aggregator)
        self.traj_generator = TrajGenerator(incremental_cfg['num_previous_frames'], incremental_cfg.get('matching_thresh', (0.5, 0.5)))
        self.incremental_cfg = incremental_cfg

    def forward_train(
        self,
        raw_points,
        raw_points_frame_inds,
        pts_xyz,
        pts_feats,
        pts_batch_idx,
        img_metas,
        proposal_list,
        seed_info,
        gt_bboxes_3d,
        gt_labels_3d
        ):

        losses = dict()

        bsz = len(img_metas)

        traj = self.traj_generator(proposal_list, seed_info, bsz)
        # traj.visualize(raw_points[0][:, :3], 'tracklet_vis.png', 'tempo_rcnn')

        proposal_list = traj.proposal_list # add some missed rois by tracking

        sample_results = self._assign_and_sample(proposal_list, gt_bboxes_3d,
                                                 gt_labels_3d)
        traj.update_after_sampling(sample_results)

        bbox_results = self._bbox_forward_train(
            pts_xyz,
            pts_feats,
            raw_points, 
            raw_points_frame_inds,
            pts_batch_idx,
            sample_results,
            traj
        )

        losses.update(bbox_results['loss_bbox'])

        return losses

    def simple_test(
        self,
        pts_xyz,
        pts_feats,
        raw_points,
        raw_points_frame_inds,
        pts_batch_inds,
        img_metas,
        proposal_list,
        seed_info,
        gt_bboxes_3d,
        gt_labels_3d,
        **kwargs):

        assert len(proposal_list) == 1, 'only support bsz==1 to make cls_preds and labels_3d consistent with bbox_results'

        traj = self.traj_generator(proposal_list, [seed_info,], 1)
        proposal_list = traj.proposal_list # add some missed rois by tracking
        traj.update_after_sampling(None) # fake the sampling process

        rois = bbox3d2roi([res[0].tensor for res in proposal_list])
        cls_preds = [res[1] for res in proposal_list]
        labels_3d = [res[2] for res in proposal_list]

        # fake in backward match, do not need it here
        # if len(rois) == 0:
        #     rois = torch.tensor([[0,0,0,5,1,1,1,0]], dtype=rois.dtype, device=rois.device)
        #     cls_preds = [torch.tensor([0.0], dtype=torch.float32, device=rois.device)]
        #     labels_3d = [torch.tensor([0], dtype=torch.int64, device=rois.device)]

        assert torch.isclose(traj.current_rois, rois).all()

           
        bbox_results = self._bbox_forward(pts_xyz, pts_feats, raw_points, raw_points_frame_inds, pts_batch_inds, rois, traj)

        bbox_list = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            bbox_results['valid_roi_mask'],
            labels_3d,
            cls_preds,
            img_metas,
            cfg=self.test_cfg)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def _bbox_forward_train(self, pts_xyz, pts_feats, raw_points, raw_points_frame_inds, batch_idx, sampling_results, traj):

        rois = bbox3d2roi([res.bboxes for res in sampling_results])
        assert torch.isclose(traj.current_rois, rois).all()
        if self.train_cfg.get('roi_aug', False):
            cfg = self.train_cfg
            assert rois.size(1) == 8
            num_rois = len(rois)

            if isinstance(cfg['xyz_noise'], (list, tuple)):
                xyz_noise = torch.tensor(cfg['xyz_noise'], dtype=rois.dtype, device=rois.device)[None, :]
            else:
                xyz_noise = cfg['xyz_noise']

            if isinstance(cfg['dim_noise'], (list, tuple)):
                dim_noise = torch.tensor(cfg['dim_noise'], dtype=rois.dtype, device=rois.device)[None, :]
            else:
                dim_noise = cfg['dim_noise']

            xyz_noise = (torch.rand((num_rois, 3), dtype=rois.dtype, device=rois.device) - 0.5) * 2 * xyz_noise
            dim_noise = (torch.rand((num_rois, 3), dtype=rois.dtype, device=rois.device) - 0.5) * 2 * dim_noise + 1
            yaw_noise = (torch.rand((num_rois,), dtype=rois.dtype, device=rois.device) - 0.5)   * 2 * cfg['yaw_noise']
            rois[:, 1:4] += xyz_noise
            rois[:, 4:-1] *= dim_noise
            rois[:, -1] += yaw_noise

        bbox_results = self._bbox_forward(pts_xyz, pts_feats, raw_points, raw_points_frame_inds, batch_idx, rois, traj)

        bbox_targets = self.bbox_head.get_targets(sampling_results, self.train_cfg)

        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            bbox_results['valid_roi_mask'],
            rois,
            *bbox_targets
        )

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, pts_xyz, pts_feats, raw_points, raw_points_frame_inds, batch_idx, rois, traj):

        assert pts_xyz.size(0) == pts_feats.size(0) == batch_idx.size(0)

        ext_pts_inds, ext_pts_roi_inds, ext_pts_info = self.roi_extractor(
            pts_xyz[:, :3], # intensity might be in pts_xyz
            batch_idx,
            rois[:, :8],
        )

        if self.traj_checkpointing:
            traj_feature, all_pre_rois, nonempty_mask = checkpoint(self.traj_feature_extractor, pts_xyz, pts_feats, raw_points, raw_points_frame_inds, batch_idx, traj, self.roi_extractor, rois)
        else:
            traj_feature, all_pre_rois, nonempty_mask = self.traj_feature_extractor(
                pts_xyz, pts_feats, raw_points, raw_points_frame_inds, batch_idx, traj, self.roi_extractor, rois) # [N_all_frames - 1, N_rois_this_batch, C]

        if traj_feature is not None:
            assert traj_feature.size(1) == len(rois)
            assert (all_pre_rois[0, :, 0] == rois[:, 0]).all() # batch_idx should match

        new_pts_feats = pts_feats[ext_pts_inds]
        new_pts_xyz = pts_xyz[ext_pts_inds]

        traj_info = dict(
            traj_feature=traj_feature,
            traj=traj,
            nonempty_mask=nonempty_mask
        )

        cls_score, bbox_pred, valid_roi_mask = self.bbox_head(
            new_pts_xyz,
            new_pts_feats,
            ext_pts_info,
            ext_pts_roi_inds,
            rois,
            traj_info,
            self.traj_feature_aggregator,
        )

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            valid_roi_mask=valid_roi_mask,
        )

        return bbox_results
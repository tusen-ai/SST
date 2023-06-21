import warnings
import torch
from torch.nn import functional as F

from mmdet3d.core import AssignResult, PseudoSampler
from mmdet3d.core.bbox import bbox3d2result, bbox3d2roi, LiDARInstance3DBoxes
from mmdet.core import build_assigner, build_sampler
from mmdet.models import HEADS
from ..builder import build_head, build_roi_extractor, build_backbone
from .base_3droi_head import Base3DRoIHead
from ipdb import set_trace



@HEADS.register_module()
class TrackletRoIHead(Base3DRoIHead):
    """Part aggregation roi head for PartA2.
    Args:
        semantic_head (ConfigDict): Config of semantic head.
        num_classes (int): The number of classes.
        seg_roi_extractor (ConfigDict): Config of seg_roi_extractor.
        part_roi_extractor (ConfigDict): Config of part_roi_extractor.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 num_classes=3,
                 roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 general_cfg=dict()):
        super().__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.general_cfg = general_cfg
        self.num_classes = num_classes
        self.with_roi_scores = general_cfg.get('with_roi_scores', False)
        self.with_roi_corners = general_cfg.get('with_roi_corners', False)
        self.checkpointing = general_cfg.get('checkpointing', False)

        self.roi_extractor = build_roi_extractor(roi_extractor)

        self.init_assigner_sampler()

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    def init_mask_head(self):
        pass

    def init_bbox_head(self, bbox_head):
        self.bbox_head = build_head(bbox_head)
        self.bbox_head.train_cfg = self.train_cfg
        self.bbox_head.test_cfg = self.test_cfg

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_sampler = PseudoSampler()
        if self.train_cfg is not None:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)

    def forward_train(
        self,
        pts_xyz,
        pts_feats,
        pts_batch_idx,
        pts_frame_inds,
        img_metas,
        tracklet_list,
        gt_candidates_list,
        ):

        losses = dict()

        sample_results = self._assign_and_sample(tracklet_list, gt_candidates_list)

        # tracklet_debug
        # boxes_in_trk = LiDARInstance3DBoxes.cat([t.concated_boxes() for t in tracklet_list])
        # boxes_in_sampled = torch.cat([s.bboxes for s in sample_results], 0)
        # try:
        #     assert torch.isclose(boxes_in_sampled, boxes_in_trk.tensor).all()
        # except AssertionError:
        #     set_trace()

        bbox_results = self._bbox_forward_train(
            pts_xyz,
            pts_feats,
            pts_batch_idx,
            pts_frame_inds,
            sample_results
        )

        losses.update(bbox_results['loss_bbox'])

        return losses

    def simple_test(
        self,
        pts_xyz,
        pts_feats,
        pts_batch_idx,
        pts_frame_inds,
        img_metas,
        tracklet_list,
        gt_candidates_list=None,
        **kwargs):

        """Simple testing forward function of PartAggregationROIHead.
        Note:
            This function assumes that the batch size is 1
        Args:
            feats_dict (dict): Contains features from the first stage.
            voxels_dict (dict): Contains information of voxels.
            img_metas (list[dict]): Meta info of each image.
            proposal_list (list[dict]): Proposal information from rpn.
        Returns:
            dict: Bbox results of one frame.
        """
        if gt_candidates_list is not None:
            gt_tracklets = self._select_one2one_candidates(tracklet_list, gt_candidates_list)
            gt_rois = self.get_gt_rois(tracklet_list, gt_tracklets)
        else:
            gt_rois = None



        rois, roi_frame_inds, cls_preds, labels_3d = self.tracklets2rois(tracklet_list)
        assert rois[:, 0].max().item() + 1 == len(tracklet_list), 'make sure there is no empty tracklet'

        bbox_results = self._bbox_forward(pts_xyz, pts_feats, pts_batch_idx, pts_frame_inds, rois, cls_preds, roi_frame_inds)

        decoded_result_list = self.bbox_head.get_bboxes_from_tracklet(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            bbox_results['valid_roi_mask'],
            labels_3d,
            cls_preds,
            img_metas,
            gt_rois=gt_rois,
            cfg=self.test_cfg,
        )

        out_tracklets = []
        assert len(decoded_result_list) == len(tracklet_list)
        for i in range(len(tracklet_list)):
            old_trk = tracklet_list[i]
            boxes, scores, labels, valid_mask = decoded_result_list[i]
            if self.test_cfg.get('tta', None) is not None:
                boxes = self.inverse_aug(old_trk, boxes, img_metas[i])
            old_trk.update_from_prediction(boxes, scores, labels, valid_mask)
            out_tracklets.append(old_trk)
        

        return out_tracklets

    def inverse_aug(self, trk, boxes, meta):
        if meta['pcd_horizontal_flip']:
            boxes.flip('horizontal')
            trk.flip('horizontal')
        if meta['pcd_vertical_flip']:
            boxes.flip('vertical')
            trk.flip('vertical')
        if 'pcd_rot_angle' in meta:
            assert trk.rot_angle == meta['pcd_rot_angle']
            boxes.rotate(-meta['pcd_rot_angle'])
            trk.rotate(-meta['pcd_rot_angle'])
        return boxes

    def _bbox_forward_train(self, pts_xyz, pts_feats, pts_batch_idx, pts_frame_inds, sampling_results):

        rois = bbox3d2roi([res.bboxes for res in sampling_results])
        roi_frame_inds = torch.cat([res.bboxes_frame_inds for res in sampling_results])
        roi_scores = torch.cat([res.scores for res in sampling_results])

        bbox_results = self._bbox_forward(pts_xyz, pts_feats, pts_batch_idx, pts_frame_inds, rois, roi_scores, roi_frame_inds)

        bbox_targets = self.bbox_head.get_targets(sampling_results, self.train_cfg)

        # tracklet_debug
        # pos_rois = bbox3d2roi([res.pos_bboxes for res in sampling_results])
        # pos_gt_bboxes = LiDARInstance3DBoxes(torch.cat([res.pos_gt_bboxes for res in sampling_results], 0))
        # reg_targets = bbox_targets[1]
        # decode_gts = self.bbox_head.decode_from_rois(pos_rois, reg_targets)
        # ********** there are some yaw flip, maybe a bug, pay attention**********

        # ***** IoU logs *******
        pos_gt_bboxes = LiDARInstance3DBoxes(torch.cat([res.pos_gt_bboxes for res in sampling_results], 0))
        decode_boxes = self.bbox_head.decode_from_rois(rois, bbox_results['bbox_pred'])
        reg_mask = bbox_targets[5] > 0
        pos_decode_boxes = LiDARInstance3DBoxes(decode_boxes[reg_mask])
        if len(pos_gt_bboxes) > 0:
            good_thresh = 0.7 if self.train_cfg['class_names'][0] == 'Car' else 0.5
            ious = LiDARInstance3DBoxes.aligned_iou_3d(pos_gt_bboxes, pos_decode_boxes)
            mean_iou = ious.mean().detach()
            num_good = (ious > good_thresh).sum().float().detach()

            pos_rois = LiDARInstance3DBoxes(rois[reg_mask, 1:])
            roi_ious = LiDARInstance3DBoxes.aligned_iou_3d(pos_gt_bboxes, pos_rois)
            mean_roi_iou = roi_ious.mean().detach()
            num_good_rois = (roi_ious > good_thresh).sum().float().detach()
        else:
            mean_iou = pts_xyz.new_ones(1).detach() * 0.5
            mean_roi_iou = pts_xyz.new_ones(1).detach() * 0.5
            num_good = pts_xyz.new_ones(1).detach()
            num_good_rois = pts_xyz.new_ones(1).detach()

        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            bbox_results['valid_roi_mask'],
            rois,
            *bbox_targets
        )

        loss_bbox['refined_iou'] = mean_iou
        loss_bbox['roi_iou'] = mean_roi_iou
        loss_bbox['num_good'] = num_good
        loss_bbox['num_good_rois'] = num_good_rois

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, pts_xyz, pts_feats, pts_batch_idx, pts_frame_inds, rois, roi_scores, roi_frame_inds):

        assert pts_xyz.size(0) == pts_feats.size(0) == pts_batch_idx.size(0) == pts_frame_inds.size(0)

        ext_pts_inds, ext_pts_roi_inds, ext_pts_info = self.roi_extractor(
            pts_xyz[:, :3], # intensity might be in pts_xyz
            pts_batch_idx,
            pts_frame_inds,
            rois[:, :8],
            roi_frame_inds,
        )

        new_pts_feats = pts_feats[ext_pts_inds]
        new_pts_xyz = pts_xyz[ext_pts_inds]

        if self.roi_extractor.combined:
            new_pts_frame_inds = pts_frame_inds[ext_pts_inds]
            roi_frame_inds_per_pts = roi_frame_inds[ext_pts_roi_inds]
            is_cur_frame = (new_pts_frame_inds == roi_frame_inds_per_pts).to(new_pts_feats.dtype)
            new_pts_feats = torch.cat([new_pts_feats, is_cur_frame.unsqueeze(1)], 1)

        if self.with_roi_scores:
            pts_scores = roi_scores[ext_pts_roi_inds]
            new_pts_feats = torch.cat([new_pts_feats, pts_scores.unsqueeze(1)], 1)
        # def forward(self, pts_xyz, pts_features, pts_info, roi_inds, rois):

        if self.with_roi_corners:
            corners = LiDARInstance3DBoxes(rois[:, 1:]).corners.to(pts_feats.dtype) # [num_rois, 8, 3]
            centers = rois[:, :3]
            corners = torch.cat([corners, centers[:, None, :]], 1)
            corners_per_pts = corners[ext_pts_roi_inds]
            offsets = corners_per_pts - new_pts_xyz[:, None, :]
            offsets = offsets.reshape(len(offsets), 27) / 10
            new_pts_feats = torch.cat([new_pts_feats, offsets], 1)


        cls_score, bbox_pred, valid_roi_mask = self.bbox_head(
            new_pts_xyz,
            new_pts_feats,
            ext_pts_info,
            ext_pts_roi_inds,
            rois,
        )

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            valid_roi_mask=valid_roi_mask,
        )

        return bbox_results

    def _assign_and_sample(self, tracklet_list, candidates_list):
        """Assign and sample proposals for training.
        Args:
            proposal_list (list[dict]): Proposals produced by RPN.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels
        Returns:
            list[:obj:`SamplingResult`]: Sampled results of each training
                sample.
        """
        assert len(tracklet_list) == len(candidates_list)
        gt_tracklet_list = self._select_one2one_candidates(tracklet_list, candidates_list)
        assert len(tracklet_list) == len(gt_tracklet_list)

        sampling_results = []
        # bbox assign
        for tid in range(len(tracklet_list)):

            trk_pd = tracklet_list[tid]
            trk_gt = gt_tracklet_list[tid]

            cur_boxes = trk_pd.concated_boxes()
            cur_gt_bboxes = trk_gt.concated_boxes()
            # cur_gt_labels = torch.full((len(cur_gt_bboxes),), trk_gt.type, device=cur_gt_bboxes.device, dtype=torch.long)

            assign_result = self.bbox_assigner.assign(trk_pd, trk_gt)
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       cur_boxes.tensor,
                                                       cur_gt_bboxes.tensor,
                                                       )
            reorder_inds = torch.cat([sampling_result.pos_inds, sampling_result.neg_inds])
            sampling_result.iou = assign_result.max_overlaps[reorder_inds].detach()
            sampling_result.bboxes_frame_inds = torch.arange(len(cur_boxes), device=trk_pd.device, dtype=torch.long)[reorder_inds]
            sampling_result.scores = assign_result.scores[reorder_inds]

            # print(sampling_result.bboxes_frame_inds, sampling_result.pos_inds)
            assert isinstance(self.bbox_sampler, PseudoSampler), 'bboxes_frame_inds is corrent only if using PseudoSampler'

            if sampling_result.pos_gt_bboxes.size(1) == 4 and self.train_cfg.get('hack_sampler_bug', False):
                assert sampling_result.pos_gt_bboxes.size(0) == 0
                sampling_result.pos_gt_bboxes = sampling_result.pos_gt_bboxes.new_zeros((0, 7))

            sampling_results.append(sampling_result)
        return sampling_results

    def _select_one2one_candidates(self, tracklet_list, candidates_list):
        candidate_thresh = self.train_cfg.get('candidate_thresh', 0.5)
        out_trks = []
        for trk, candidates in zip(tracklet_list, candidates_list):
            if len(candidates) == 0:
                out_trks.append(trk.new_empty())
                continue
            affinities = torch.tensor([(trk.intersection_ious(c) > candidate_thresh).sum() for c in candidates])
            if self.train_cfg.get('merge_candidates', False):
                merged_candidate = self._merge_candidates(candidates, affinities)
                out_trks.append(merged_candidate)
            else:
                argmax = torch.argmax(affinities).item()
                out_trks.append(candidates[argmax])
        return out_trks

    def _merge_candidates(self, candidates, priority):
        candidates_len = [len(c) for c in candidates]
        candidates = [first for first, second in sorted(zip(candidates, priority), key=lambda pair: -pair[1])]
        new_len = [len(c) for c in candidates]
        base = candidates[0]
        for c in candidates[1:]:
            base.merge_not_exist(c)
        return base

    def tracklets2rois(self, tracklets):
        rois = bbox3d2roi([t.concated_boxes().tensor for t in tracklets])
        cls_preds = torch.cat([t.concated_scores() for t in tracklets])
        labels_3d = torch.cat([t.concated_labels() for t in tracklets])

        roi_frame_inds = torch.cat([torch.arange(len(t), device=rois.device, dtype=torch.long) for t in tracklets])

        assert tracklets[0].type_format == 'mmdet3d'
        assert (labels_3d <= 2).all(), 'Holds in WOD'
        return rois, roi_frame_inds, cls_preds, labels_3d

    def convert_result_to_tracklet(self, tracklet, result):
        bboxes, scores, labels = result

    def get_gt_rois(self, tracklets, gt_tracklets):
        assert len(tracklets) == len(gt_tracklets)
        out_boxes_list = []
        out_mask_list = []
        for i in range(len(tracklets)):
            trk = tracklets[i]
            gt = gt_tracklets[i]
            boxes, mask = gt.concated_boxes_from_ts(trk.ts_list)
            out_boxes_list.append(boxes)
            out_mask_list.append(mask)
        gt_rois = torch.cat(out_boxes_list, 0)
        mask = torch.cat(out_mask_list, 0)
        gt_rois = torch.cat([mask[:, None], gt_rois], 1)
        return gt_rois
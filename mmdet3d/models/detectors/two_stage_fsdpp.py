from .single_stage_fsd import SingleStageFSD
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d, LiDARInstance3DBoxes
from ipdb import set_trace

from mmdet.models import build_detector

from .. import builder
from mmdet3d.ops import scatter_v2, get_inner_win_inds
from mmdet.core import multi_apply

from .incremental_ops import (find_delta_points_by_voxelization, box_frame_transform_gpu,
    find_delta_points_by_voxelization_list, points_frame_transform, generate_virtual_seed_points,
    find_delta_points_by_voxelization_list_v2,
    find_delta_points_by_voxelization_list_v3,
    )


import numpy as np
from mmdet3d.datasets.waymo_dataset import box_frame_transform
import time
import copy

try:
    from torchex import group_fps
except ImportError:
    group_fps = None

from mmdet3d.utils import TorchTimer
timer = TorchTimer(-1)

@DETECTORS.register_module()
class TwoStageFSDPP(SingleStageFSD):

    def __init__(self,
                 backbone,
                 segmentor,
                 voxel_layer=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 neck=None,
                 bbox_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 cluster_assigner=None,
                 pretrained=None,
                 init_cfg=None,
                 incremental_cfg=None):
        super().__init__(
            backbone=backbone,
            segmentor=segmentor,
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            cluster_assigner=cluster_assigner,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

        # update train and test cfg here for now
        rcnn_train_cfg = train_cfg.rcnn if train_cfg else None
        roi_head.update(train_cfg=rcnn_train_cfg)
        roi_head.update(test_cfg=test_cfg.rcnn)
        roi_head.pretrained = pretrained
        self.roi_head = builder.build_head(roi_head)
        self.num_classes = self.bbox_head.num_classes
        self.runtime_info = dict()
        self.incremental_cfg = incremental_cfg
        self.max_pre_frames = incremental_cfg.get('num_previous_frames', 4)
        print(f'Use {self.max_pre_frames} previous frames')
        self.rcnn_type = roi_head['type']
        assert self.rcnn_type in ['GroupCorrectionHead', 'IncrementalROIHead']
        self.iters = 0

        self.last_segind='None'
        self.previous_pc=[]
        self.previous_delta_pc = []
        self.previous_pose=[]
        self.previous_seed_info = []
        self.previous_cropped_pc = []
        self.frame_counter = 0

    def forward_train(self,
                      points,
                      pts_frame_inds,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      seed_info=None,
                      gt_bboxes_ignore=None):

        seed_info = self.preprocess_seed(seed_info)

        gt_bboxes_3d = [b[l>=0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d = [l[l>=0] for l in gt_labels_3d]

        new_points, num_delta_points_list = self.generate_points(points, pts_frame_inds, seed_info)


        point_drop_ratio = self.train_cfg.get('point_drop_ratio', 0)
        if point_drop_ratio > 0:
            tmp_list = []
            for p in new_points:
                idx = torch.randperm(len(p)).to(p.device) # bug in torch1.8
                keep_num = int(len(p) * (1-point_drop_ratio))
                tmp_list.append(p[idx[:keep_num]])
            new_points = tmp_list



        self.print_info['num_input_points'] = new_points[0].new_zeros(1) + sum([len(p) for p in new_points]) / len(new_points)
        self.print_info['num_delta_points'] = new_points[0].new_zeros(1) + sum(num_delta_points_list) / len(new_points)
        # self.vis_delta_points(points[2], pts_frame_inds[2], new_points[2], num_delta_points_list[2])
        # vis_bev_pc(torch.cat(pre_points[2] + [points[2],], 0), name='multi_frames.png', dir='incremental')
        # set_trace()

        losses = {}
        rpn_outs = super().forward_train(
            points=new_points,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_ignore=gt_bboxes_ignore,
            runtime_info=self.runtime_info
        )
        losses.update(rpn_outs['rpn_losses'])

        proposal_list = self.bbox_head.get_bboxes(
            rpn_outs['cls_logits'], rpn_outs['reg_preds'], rpn_outs['cluster_xyz'], rpn_outs['cluster_inds'], img_metas
        )

        assert len(proposal_list) == len(gt_bboxes_3d) # make sure the length is batch size

        pts_xyz, pts_feats, pts_batch_inds = self.prepare_multi_class_roi_input(
            rpn_outs['all_input_points'],
            rpn_outs['valid_pts_feats'],
            rpn_outs['seg_feats'],
            rpn_outs['pts_mask'],
            rpn_outs['pts_batch_inds'],
            rpn_outs['valid_pts_xyz']
        )

        # shuffle point. I forget why the shuffle is needed 
        # inds = torch.randperm(pts_xyz.shape[0])
        # pts_xyz = pts_xyz[inds].contiguous()
        # pts_feats = pts_feats[inds].contiguous()
        # pts_batch_inds = pts_batch_inds[inds].contiguous()

        if self.rcnn_type == 'IncrementalROIHead':
            roi_losses = self.roi_head.forward_train(
                points, 
                pts_frame_inds,
                pts_xyz,
                pts_feats,
                pts_batch_inds,
                img_metas,
                proposal_list,
                seed_info,
                gt_bboxes_3d,
                gt_labels_3d,
            )
        else:
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

    def prepare_roi_input(self, points, cluster_pts_feats, pts_seg_feats, pts_mask, pts_batch_inds, cluster_pts_xyz):
        assert isinstance(pts_mask, list)
        pts_mask = pts_mask[0]
        assert points.shape[0] == pts_seg_feats.shape[0] == pts_mask.shape[0] == pts_batch_inds.shape[0]

        if self.training and self.train_cfg.get('detach_seg_feats', False):
            pts_seg_feats = pts_seg_feats.detach()

        if self.training and self.train_cfg.get('detach_cluster_feats', False):
            cluster_pts_feats = cluster_pts_feats.detach()

        pad_feats = cluster_pts_feats.new_zeros(points.shape[0], cluster_pts_feats.shape[1])
        pad_feats[pts_mask] = cluster_pts_feats
        assert torch.isclose(points[pts_mask], cluster_pts_xyz).all()

        cat_feats = torch.cat([pad_feats, pts_seg_feats], dim=1)

        return points, cat_feats, pts_batch_inds

    def prepare_multi_class_roi_input(self, points, cluster_pts_feats, pts_seg_feats, pts_mask, pts_batch_inds, cluster_pts_xyz):
        assert isinstance(pts_mask, list)
        bg_mask = sum(pts_mask) == 0
        assert points.shape[0] == pts_seg_feats.shape[0] == bg_mask.shape[0] == pts_batch_inds.shape[0]

        if self.training and self.train_cfg.get('detach_seg_feats', False):
            pts_seg_feats = pts_seg_feats.detach()

        if self.training and self.train_cfg.get('detach_cluster_feats', False):
            cluster_pts_feats = cluster_pts_feats.detach()


        ##### prepare points for roi head
        fg_points_list = [points[m] for m in pts_mask]
        all_fg_points = torch.cat(fg_points_list, dim=0)

        assert torch.isclose(all_fg_points, cluster_pts_xyz).all()

        bg_pts_xyz = points[bg_mask]
        all_points = torch.cat([bg_pts_xyz, all_fg_points], dim=0)
        #####

        ##### prepare features for roi head
        fg_seg_feats_list = [pts_seg_feats[m] for m in pts_mask]
        all_fg_seg_feats = torch.cat(fg_seg_feats_list, dim=0)
        bg_seg_feats = pts_seg_feats[bg_mask]
        all_seg_feats = torch.cat([bg_seg_feats, all_fg_seg_feats], dim=0)

        num_out_points = len(all_points)
        assert num_out_points == len(all_seg_feats)

        pad_feats = cluster_pts_feats.new_zeros(bg_mask.sum(), cluster_pts_feats.shape[1])
        all_cluster_pts_feats = torch.cat([pad_feats, cluster_pts_feats], dim=0)
        #####

        ##### prepare batch inds for roi head
        bg_batch_inds = pts_batch_inds[bg_mask]
        fg_batch_inds_list = [pts_batch_inds[m] for m in pts_mask]
        fg_batch_inds = torch.cat(fg_batch_inds_list, dim=0)
        all_batch_inds = torch.cat([bg_batch_inds, fg_batch_inds], dim=0)


        # pad_feats[pts_mask] = cluster_pts_feats

        cat_feats = torch.cat([all_cluster_pts_feats, all_seg_feats], dim=1)

        # sort for roi extractor
        all_batch_inds, inds = all_batch_inds.sort()
        all_points = all_points[inds]
        cat_feats = cat_feats[inds]

        return all_points, cat_feats, all_batch_inds

    def simple_test(self, points, img_metas, imgs=None, pts_frame_inds=None, seed_info=None, pose=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):

        assert len(seed_info) == 1
        seed_info = seed_info[0]
        seed_info = self.preprocess_seed(seed_info)

        if self.test_cfg.get('reuse_test', False):
            assert self.test_cfg['sequential']
            with timer.timing('Total FSDPP'):
                return self.reuse_simple_test(points, img_metas, imgs, pts_frame_inds, seed_info, pose, rescale, gt_bboxes_3d, gt_labels_3d)

        # I don't know why seed_info is wrapped by one more list than in forward_train
        # Maybe the forward_test in base detector has indexed the first element for each positional arguements, 
        # but not do this for kwargs.
        raise NotImplementedError('Need to change the data config, adding LoadPreviousFramesWaymo')
        pts_frame_inds = pts_frame_inds[0]
        new_points, num_delta_points_list = self.generate_points(points, pts_frame_inds, seed_info)
        assert len(new_points[0]) > 0

        return self.input_agnostic_simple_test(new_points, img_metas, gt_bboxes_3d, gt_labels_3d)

    def input_agnostic_simple_test(self, points, img_metas, seed_info, gt_bboxes_3d=None, gt_labels_3d=None, raw_points=None, raw_points_frames_inds=None):

        # if self.test_cfg.get('extract_noisy_fg_at_first', False):
        #     points = self.extract_fg_by_gt(points, gt_bboxes_3d, gt_labels_3d, self.test_cfg['extract_fg_extra_width'])

        with timer.timing('rpn'):
            rpn_outs = super().simple_test(
                points=points,
                img_metas=img_metas,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
            )

        # proposal_list = self.bbox_head.get_bboxes(
        #     rpn_outs['cls_logits'], rpn_outs['reg_preds'], rpn_outs['cluster_xyz'], rpn_outs['cluster_inds'], img_metas
        # )
        with timer.timing('between rpn and rcnn'):
            proposal_list = rpn_outs['proposal_list']

            if self.test_cfg.get('skip_rcnn', False):
                bbox_results = [
                    bbox3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels in proposal_list
                ]
                return bbox_results

            if self.num_classes > 1 or self.test_cfg.get('enable_multi_class_test', False):
                prepare_func = self.prepare_multi_class_roi_input
            else:
                prepare_func = self.prepare_roi_input

            pts_xyz, pts_feats, pts_batch_inds = prepare_func(
                rpn_outs['all_input_points'],
                rpn_outs['valid_pts_feats'],
                rpn_outs['seg_feats'],
                rpn_outs['pts_mask'],
                rpn_outs['pts_batch_inds'],
                rpn_outs['valid_pts_xyz']
            )

            # shuffle point
            # inds = torch.randperm(pts_xyz.shape[0]).to(pts_xyz.device)
            # pts_xyz = pts_xyz[inds].contiguous()
            # pts_feats = pts_feats[inds].contiguous()
            # pts_batch_inds = pts_batch_inds[inds].contiguous()

        with timer.timing('rcnn'):
            if self.rcnn_type == 'IncrementalROIHead':
                results = self.roi_head.simple_test(
                    pts_xyz,
                    pts_feats,
                    raw_points,  # place holder for raw points
                    raw_points_frames_inds,  # place holder for raw points
                    pts_batch_inds,
                    img_metas,
                    proposal_list,
                    seed_info,
                    gt_bboxes_3d,
                    gt_labels_3d,
                )
            else:
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

    def reuse_simple_test(self, points, img_metas, imgs=None, pts_frame_inds=None, seed_info=None, pose=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):

        assert len(seed_info) == 1
        seed_info = seed_info[0]
        assert pts_frame_inds is None

        cur_points = points

        sample_idx = img_metas[0]['sample_idx']
        segment_idx = f'{sample_idx:07d}'[1:4]
        reset=(self.last_segind != segment_idx)
        if reset:
            self.previous_pc = []
            self.previous_sample_idx = []
            self.previous_pose = []
            self.previous_seed_info = []
            self.previous_cropped_pc = []
            self.frame_counter = 0
            self.previous_delta_pc = []

        curr_pose = pose[0].squeeze().float()

        seed_info = self.get_seed_info(img_metas, seed_info, curr_pose, cur_points[0].device) # transform previous seed info or previous results to uniform seed info
        with timer.timing('filter_and_merge_pc'):
            merged_pc = self.filter_and_merge_pc(cur_points, segment_idx, curr_pose, seed_info)  

        results = self.input_agnostic_simple_test([merged_pc,], img_metas, seed_info, gt_bboxes_3d, gt_labels_3d)

        with timer.timing('Finishing'):
            cur_seed = self.result2seed(results, cur_points[0].device)
            self.previous_seed_info.append(cur_seed)
            self.frame_counter += 1

            cur_modified_seed = self.modify_previous_boxes(cur_seed, cur_points[0].device)

            with timer.timing('crop_points'):
                cur_cropped_pc = self.crop_and_process_points(cur_points[0], cur_modified_seed)

            self.previous_cropped_pc.append(cur_cropped_pc)

            if len(self.previous_pc) > self.max_pre_frames:
                self.previous_pc.pop(0)
                self.previous_sample_idx.pop(0)
                self.previous_pose.pop(0)
                self.previous_seed_info.pop(0)
                self.previous_cropped_pc.pop(0)

            self.last_segind=segment_idx

        return results

    def filter_and_merge_pc(self, points, sample_idx, curr_pose, seed_info):

        assert len(self.previous_pc) <= self.max_pre_frames

        points=points[0]

        # previous_pc_incurr=copy.deepcopy(self.previous_pc)
        # previous_pc_incurr=[e.clone() for e in self.previous_pc]
        previous_pc_incurr = []

        curr_pose_inv = torch.linalg.inv(curr_pose)
        for i, pc in enumerate(self.previous_pc):
            past_pc_in_curr = points_frame_transform(pc[:, :3], self.previous_pose[i], None, curr_pose_inv)
            # previous_pc_incurr[i][:,:3] = past_pc_in_curr
            previous_pc_incurr.append(torch.cat([past_pc_in_curr, pc[:, 3:]], 1))

        out_points_list, num_delta_points_list = self.generate_points_list(points, previous_pc_incurr, seed_info, curr_pose)


        new_points = out_points_list[0]

        # temp for point drop test
        # shuffle_idx = torch.randperm(len(new_points)).to(new_points.device) # bug in torch1.8
        # new_points = new_points[shuffle_idx]

        self.previous_pc.append(points)
        self.previous_pose.append(curr_pose)
        self.previous_sample_idx.append(sample_idx)
        assert len(set(self.previous_sample_idx))==1
        return new_points

    def extract_fg_by_gt(self, point_list, gt_bboxes_3d, gt_labels_3d, extra_width):
        if isinstance(gt_bboxes_3d[0], list):
            assert len(gt_bboxes_3d) == 1
            gt_bboxes_3d = gt_bboxes_3d[0]

        bsz = len(point_list)

        new_point_list = []
        for i in range(bsz):
            points = point_list[i]
            gts = gt_bboxes_3d[i].to(points.device)
            if len(gts) == 0:
                this_fg_mask = points.new_zeros(len(points), dtype=torch.bool)
                this_fg_mask[:min(1000, len(points))] = True
            else:
                enlarged_gts = gts.enlarged_box(extra_width)
                pts_inds = enlarged_gts.points_in_boxes(points[:, :3])
                this_fg_mask = pts_inds > -1
                if not this_fg_mask.any():
                    this_fg_mask[:min(1000, len(points))] = True

            new_point_list.append(points[this_fg_mask])
        return new_point_list


    def generate_points(self, points, pts_frame_inds, seed_info):

        out_points_list = []
        num_delta_points_list = []
        bsz = len(points)

        for i in range(bsz):

            cur_mask = pts_frame_inds[i] == 0
            pre_mask = (pts_frame_inds[i] < 0) & (pts_frame_inds[i] >= -self.max_pre_frames)
            cur_points = points[i][cur_mask]
            pre_points = points[i][pre_mask]
            pre_frame_inds = pts_frame_inds[i][pre_mask]

            old_points = self.get_old_points(pre_points, pre_frame_inds, seed_info[i])
            if self.incremental_cfg.get('disable_incremental', False):
                this_new_points = old_points
            else:

                num_base_frame = self.incremental_cfg.get('num_base_frame', self.max_pre_frames)
                base_mask = pre_frame_inds >= -num_base_frame

                delta_points = self.get_delta_points(cur_points, pre_points[base_mask])
                delta_points = F.pad(delta_points, (0, 1), 'constant', 0) # hard code frame IDs

                max_age = self.incremental_cfg.get('max_age', 0)
                # assert max_age in (0, 1), 'only support one more delta points'
                elegant_max_age = self.incremental_cfg.get('elegant_max_age', False)
                if max_age > 0 and not elegant_max_age:
                    last_mask = pre_frame_inds == -1
                    last_delta_points = self.get_delta_points(pre_points[last_mask], pre_points[(~last_mask) & base_mask])
                    last_delta_points = F.pad(last_delta_points, (0, 1), 'constant', -0.1) # hard code frame IDs
                    delta_points = torch.cat([delta_points, last_delta_points], 0)
                elif max_age > 0 and elegant_max_age:
                    previous_delta_points = self.get_previous_delta_points_by_max_age_training(
                        points[i], pts_frame_inds[i]
                    )
                    delta_points = torch.cat([delta_points, previous_delta_points], 0)

                num_delta_points_list.append(len(delta_points))
                this_new_points = torch.cat([old_points, delta_points], dim=0)
            out_points_list.append(this_new_points)

        return out_points_list, num_delta_points_list

    def get_previous_delta_points_by_max_age_training(self, points, pts_inds):
        """
        Here the pre_points contains all loaded previous points, including those excess the num_pre_frames.
        """
        assert self.training
        max_age = self.incremental_cfg['max_age']
        num_base_frame = self.incremental_cfg.get('num_base_frame', self.max_pre_frames)
        assert num_base_frame + max_age + 1 <= 8, 'Currently we only load 8 frames'
        pre_delta_points_list = []
        for i in range(max_age):
            this_age = i+1
            frame_idx = -this_age
            inc_mask = pts_inds == frame_idx
            base_mask = (pts_inds >= -(num_base_frame + this_age)) & (pts_inds < -this_age)
            assert not (inc_mask & base_mask).any()
            assert base_mask.max().item() - base_mask.min().item() <= num_base_frame
            this_delta_points = self.get_delta_points(points[inc_mask], points[base_mask])
            this_delta_points = F.pad(this_delta_points, (0, 1), 'constant', -this_age/10) # hard code frame IDs
            pre_delta_points_list.append(this_delta_points)
        pre_delta_points = torch.cat(pre_delta_points_list, 0)
        return pre_delta_points


    def generate_points_list(self, cur_points, pre_points_list, seed_info, cur_pose):
        assert not self.training, 'only used in single-sample testing'
        if len(seed_info) > len(pre_points_list):
            pre_points_list.append(cur_points)
            assert len(seed_info) == len(pre_points_list), f'{len(seed_info)} {len(pre_points_list)}'

        old_points = self.get_old_points_list_v2(cur_points, seed_info, cur_pose)
        if self.incremental_cfg.get('disable_incremental', False):
            this_new_points = old_points
            num_delta_points = 0
        else:
            num_base_frame = self.incremental_cfg.get('num_base_frame', self.max_pre_frames)
            delta_points = self.get_delta_points_list(cur_points, pre_points_list[-num_base_frame:])
            num_delta_points = len(delta_points)
            delta_points = F.pad(delta_points, (0, 1), 'constant', 0) # hard code frame IDs

            max_age = self.incremental_cfg.get('max_age', 0)
            # assert max_age in (0, 1), 'only support one more delta points'
            elegant_max_age = self.incremental_cfg.get('elegant_max_age', False)
            if max_age > 0 and not elegant_max_age:
                last_delta_points = self.get_delta_points_list(pre_points_list[-1], pre_points_list[-num_base_frame:-1])
                last_delta_points = F.pad(last_delta_points, (0, 1), 'constant', -0.1) # hard code frame IDs
                delta_points = torch.cat([delta_points, last_delta_points], 0)
            elif max_age > 0 and elegant_max_age:

                previous_delta_points = self.get_previous_delta_points_by_max_age_test(cur_pose)

                # enqueue and  dequeue
                if len(self.previous_delta_pc) == max_age:
                    self.previous_delta_pc.pop(0)
                self.previous_delta_pc.append(delta_points) # first add current delta_points to list

                if previous_delta_points is not None:
                    delta_points = torch.cat([delta_points, previous_delta_points], 0)

            this_new_points = torch.cat([old_points, delta_points], dim=0)

        return [this_new_points,], [num_delta_points,]

    def get_previous_delta_points_by_max_age_test(self, cur_pose):

        assert not self.training
        max_age = self.incremental_cfg['max_age']
        pre_delta_list = []
        exsiting_num = len(self.previous_delta_pc)
        assert exsiting_num <= max_age
        if exsiting_num == 0:
            print('\nNo previous points now, skip...')

        for i in range(exsiting_num):
            this_age = i+1
            pc = self.previous_delta_pc[-this_age]
            pre_pose = self.previous_pose[-this_age]
            pc_in_curr = points_frame_transform(pc[:, :3], pre_pose, cur_pose)
            pc_in_curr = torch.cat([pc_in_curr[:, :3], pc[:, 3:]], 1)
            assert (pc_in_curr[:, -1] == 0).all()
            pc_in_curr[:, -1] = -this_age/10
            pre_delta_list.append(pc_in_curr)
        if len(pre_delta_list) > 0:
            pre_delta_points = torch.cat(pre_delta_list, 0)
            return pre_delta_points
        else:
            return None

    def get_old_points(self, pre_points, pre_frame_inds, frames_seed_info):
        """
        coordinates of points and boxes are all transformed to the current frame.
        """

        assert isinstance(frames_seed_info, list), 'frame-wise list'
        num_pre_frames = len(frames_seed_info)

        if not num_pre_frames == -1 * pre_frame_inds.min().item():
            set_trace()

        assert num_pre_frames == -1 * pre_frame_inds.min().item()
        assert num_pre_frames >= 1
        device = pre_points.device

        select_points_list = []

        if  'crop_frames' in self.incremental_cfg:
            num_pre_frames = min(self.incremental_cfg['crop_frames'], num_pre_frames)

        for i in range(num_pre_frames):
            this_frame_seed_info = self.modify_previous_boxes(frames_seed_info[i], device)
            this_gts = this_frame_seed_info['gt_bboxes_3d']

            if len(this_gts) == 0:
                continue

            this_frame_mask = pre_frame_inds == -i-1
            this_points = pre_points[this_frame_mask]

            select_points = self.crop_and_process_points(this_points, this_frame_seed_info)
            if len(select_points) == 0:
                continue

            select_points = F.pad(select_points, (0, 1), 'constant', float(- i - 1) / 10) # hard code frame IDs
            select_points_list.append(select_points)

        if len(select_points_list) == 0:
            out_points = pre_points[1:200, :]
            out_points = self.channel_padding(out_points)
            out_points = F.pad(out_points, (0, 1), 'constant', float(- num_pre_frames) / 10) # hard code frame IDs
        else:
            out_points = torch.cat(select_points_list, dim=0)
        return out_points

    def crop_and_process_points(self, points, seed_info):
            # inbox_inds = get_inner_win_inds(box_inds)
        boxes = seed_info['gt_bboxes_3d']
        cfg = self.incremental_cfg
        box_inds = boxes.points_in_boxes(points[:, :3])
        mask = box_inds > -1
        crop_points = points[mask]
        pos_box_inds = box_inds[mask]

        # small points in group_FPS might lead to unknown bug
        if 'n_fps' in cfg:
            if mask.sum().item() < 200:
                crop_points = self.channel_padding(crop_points)
                return crop_points

        if not mask.any():
            crop_points = self.channel_padding(crop_points)
            return crop_points


        if 'max_crop_points' in cfg:
            assert 'n_fps' not in cfg
            max_n = cfg['max_crop_points']

            if cfg.get('crop_shuffle', False):
                shuffle_inds = torch.randperm(len(pos_box_inds)).to(pos_box_inds.device)
                pos_box_inds = pos_box_inds[shuffle_inds]
                crop_points = crop_points[shuffle_inds]

            inbox_inds = get_inner_win_inds(pos_box_inds.long())
            valid_mask = inbox_inds < max_n
            crop_points = crop_points[valid_mask]
            pos_box_inds = pos_box_inds[valid_mask]
        elif 'n_fps' in cfg:
            crop_points = group_fps(crop_points, pos_box_inds, cfg['n_fps'], True)

        elif 'voxel_pooling_size' in cfg:
            crop_points = self.voxel_skeleton_sampling(crop_points, pos_box_inds, cfg['voxel_pooling_size'])



        return crop_points


    def channel_padding(self, points):
        """
        pad more channels to keep dimension same with old_points
        """ 
        assert points.size(1) == 5
        if self.incremental_cfg.get('crop_painting', False):
            points = F.pad(points, (0, 2), 'constant', -1)

        virtual_cfg = self.incremental_cfg.get('virtual_seed_points_cfg', None)
        if virtual_cfg is not None:
            points = F.pad(points, (0, virtual_cfg['padding_dim']), 'constant', 0)

        return points


    def get_old_points_list_v2(self, cur_points, frames_seed_info, cur_pose):
        """
        coordinates of points and boxes are all transformed to the current frame.
        """
        device = cur_points.device

        pre_crop_pc = []
        assert len(self.previous_cropped_pc) == len(self.previous_pose)
        cur_pose_inv = torch.linalg.inv(cur_pose)
        for pc, pre_pose in zip(self.previous_cropped_pc, self.previous_pose):
            pc_in_curr = points_frame_transform(pc[:, :3], pre_pose, None, cur_pose_inv)
            pre_crop_pc.append(torch.cat([pc_in_curr[:, :3], pc[:, 3:]], 1))

        if len(frames_seed_info) > len(pre_crop_pc):
            # this would happen num_previous_frames times in the warmup stage per sequence
            padding_seed = self.modify_previous_boxes(frames_seed_info[0], device)
            pre_crop_pc.append(self.crop_and_process_points(cur_points, padding_seed))
            assert len(frames_seed_info) == len(pre_crop_pc)

        if  'crop_frames' in self.incremental_cfg:
            crop_frames = min(self.incremental_cfg['crop_frames'], len(pre_crop_pc))
            pre_crop_pc = pre_crop_pc[-crop_frames:]

        out_pc_list = []
        for i, pc in enumerate(pre_crop_pc):
            this_out_pc = F.pad(pc, (0, 1), 'constant', float(i - len(pre_crop_pc)) / 10) # hard code frame IDs
            out_pc_list.append(this_out_pc)

        try:
            out_points = torch.cat(out_pc_list, dim=0)
        except Exception:
            print([pc.shape for pc in out_pc_list])
            raise ValueError


        if len(out_points) < 500:
            # too small numer of points may lead to error in spconv
            filler = F.pad(cur_points[:500, :], (0, 1), 'constant', 0) # hard code frame IDs
            out_points = torch.cat([filler, out_points], 0)

        return out_points

    def get_delta_points(self, cur_points, pre_points):
        cfg = self.incremental_cfg
        if len(pre_points) > 0:
            delta_points = find_delta_points_by_voxelization(pre_points, cur_points, cfg['voxel_size'], cfg['point_cloud_range'])
        else:
            delta_points = cur_points.new_zeros((0, cur_points.size(1)))
        delta_points = self.channel_padding(delta_points) 
        return delta_points

    def get_delta_points_list(self, cur_points, pre_points_list):
        cfg = self.incremental_cfg
        # delta_points = find_delta_points_by_voxelization_list(pre_points_list, cur_points, cfg['voxel_size'], cfg['point_cloud_range'])
        if len(pre_points_list) > 0:
            delta_points = find_delta_points_by_voxelization_list_v3(pre_points_list, cur_points, cfg['voxel_size'], cfg['point_cloud_range'])
        else:
            delta_points = cur_points.new_zeros((0, cur_points.size(1)))
        delta_points = self.channel_padding(delta_points)
        return delta_points

    def remove_ground(self, points):
        cfg = self.incremental_cfg
        z = points[:, 2]
        valid_mask = (z < -0.2) | (z > 0.2)
        return points[valid_mask]

    def modify_previous_boxes(self, seed_info, device):
        if len(seed_info['gt_bboxes_3d']) == 0:
            return seed_info

        cfg = self.incremental_cfg
        new_seed = {}

        boxes = seed_info['gt_bboxes_3d'].to(device)
        new_seed['origin_bboxes'] = boxes.clone()

        # add training noise
        if self.training:
            boxes = boxes.noisy_box(cfg['center_noise'], cfg['dim_noise'], cfg['yaw_noise'])

        gt_labels_3d = seed_info['gt_labels_3d']
        scores = seed_info['scores']
        if isinstance(scores, np.ndarray):
            gt_labels_3d = torch.from_numpy(gt_labels_3d).to(device).float()
            scores = torch.from_numpy(scores).to(device).float()
        else:
            gt_labels_3d = gt_labels_3d.clone()
            scores = scores.clone()
        new_seed['gt_labels_3d'] = gt_labels_3d
        new_seed['scores'] = scores

        # enlarge boxes
        extra_width = cfg['extra_width']
        if isinstance(extra_width, dict):
            boxes = boxes.classwise_enlarged_box(extra_width, gt_labels_3d)
        else:
            boxes = boxes.enlarged_box(extra_width)

        new_seed['gt_bboxes_3d'] = boxes


        return new_seed


    def vis_previous_data(self, points_list, seed_info_list):
        for i in range(len(points_list)):
            points = points_list[i]
            boxes = seed_info_list[i]['gt_bboxes_3d']
            vis_bev_pc(points, boxes, name=f'test_inc_{i}.png', dir='incremental')

    # self.vis_delta_points(points[1], pts_frame_inds[1], new_points[1], num_delta_points_list[1])
    def vis_delta_points(self, all_points, frame_inds, new_points, num_delta):
        all_pre_points = all_points[frame_inds != 0]
        delta_points = new_points[-num_delta:]
        pre_fg_points = new_points[:len(new_points) - num_delta]
        vis_bev_pc_list([all_pre_points, new_points], name='pre_and_new_points.png', dir='incremental')
        vis_bev_pc_list([pre_fg_points, delta_points], name='pre_fg_and_delta_points.png', dir='incremental')
        vis_bev_pc_list([new_points, ], name='new_points.png', dir='incremental')
        set_trace()

    def result2seed(self, results, device):
        seed = {}
        seed['gt_bboxes_3d'] = results[0]['boxes_3d'].to(device)
        seed['gt_labels_3d'] = results[0]['labels_3d'].to(device)
        seed['scores'] = results[0]['scores_3d'].float().to(device)
        return seed

    def preprocess_seed(self, seed_list, device=None):


        score_thr = self.incremental_cfg.get('pre_score_thr', 0)
        if score_thr > 0:
            if not self.training:
                assert score_thr == self.test_cfg['rcnn']['score_thr'], 'training-testing consistency'
            seed_list = [self.filter_seed_by_score(s) for s in seed_list]

        nms_thr = self.incremental_cfg.get('pre_nms_thr', 0)
        if nms_thr > 0:
            seed_list = [self.filter_seed_by_nms(s) for s in seed_list]

        seed_list = [s[:self.max_pre_frames] for s in seed_list]

        noise_cfg = self.incremental_cfg.get('noise_cfg', None)
        if noise_cfg is not None:
            assert False, 'in case of wrong config'

            seed_list = [self.random_drop_seed(s, noise_cfg.get('drop_rate', None), device) for s in seed_list]
            seed_list = [self.random_fp_insertion(s, noise_cfg.get('fp_rate', None)) for s in seed_list]

        return seed_list

    def filter_seed_by_score(self, seed_all_frames):
        new_list = []
        for seed in seed_all_frames:
            scores = seed['scores']
            if len(scores) == 0:
                new_list.append(seed)
                continue
            mask = scores > self.incremental_cfg['pre_score_thr']
            new_seed = {k:v[mask] for k, v in seed.items()}
            new_list.append(new_seed)
        return new_list

    def random_fp_insertion(self, seed_all_frames, fp_rate):
        if fp_rate is None:
            return seed_all_frames
        new_list = []

        for i, seed in enumerate(seed_all_frames):
            this_boxes = seed['gt_bboxes_3d']
            copy_mask = torch.rand(len(this_boxes.tensor)) < fp_rate
            mask_np = copy_mask.numpy()
            copy_boxes = this_boxes[copy_mask].clone()
            copy_boxes.tensor[:, :2] += (torch.rand((len(copy_boxes), 2)) - 0.5) * 20

            copy_labels = seed['gt_labels_3d'][mask_np]
            copy_scores = seed['scores'][mask_np]

            new_seed = {}
            new_seed['gt_bboxes_3d'] = LiDARInstance3DBoxes.cat([this_boxes, copy_boxes])
            new_seed['gt_labels_3d'] = np.concatenate([seed['gt_labels_3d'], copy_labels], 0)
            new_seed['scores'] = np.concatenate([seed['scores'], copy_scores], 0)

            new_list.append(new_seed)

        return new_list



        for i, seed in enumerate(seed_all_frames):
            mask = keep_mask_list[i].numpy()
            new_seed = {k:v[mask] for k, v in seed.items()}
            new_list.append(new_seed)

        # self.vis_seed_boxes(seed_all_frames)
        # set_trace()

        # if len(keep_mask_list) > 1:
        #     set_trace()

        return new_list

    def random_drop_seed(self, seed_all_frames, drop_rate, device):
        if drop_rate is None:
            return seed_all_frames
        new_list = []
        keep_mask_list = []
        base_box = seed_all_frames[0]['gt_bboxes_3d'].to(device)
        base_keep_mask = torch.rand(len(base_box), device=device) > drop_rate
        keep_mask_list.append(base_keep_mask)

        for i, seed in enumerate(seed_all_frames[1:]):
            this_boxes = seed['gt_bboxes_3d'].to(device)
            keep_mask = torch.ones(len(this_boxes.tensor), dtype=torch.bool, device=device)

            if len(this_boxes) == 0 or len(base_box) == 0:
                keep_mask_list.append(keep_mask)
                base_keep_mask = keep_mask
                base_box = this_boxes
                continue

            iou_mat = LiDARInstance3DBoxes.overlaps(base_box, this_boxes) #[N_first, N_this]
            max_iou_per_this, max_inds = iou_mat.max(0)
            matched_mask = max_iou_per_this > 0.3
            match_inds = max_inds[matched_mask]
            keep_mask[matched_mask] = base_keep_mask[match_inds]
            keep_mask_list.append(keep_mask)

            base_keep_mask = keep_mask
            base_box = this_boxes


        for i, seed in enumerate(seed_all_frames):
            mask = keep_mask_list[i].numpy()
            new_seed = {k:v[mask] for k, v in seed.items()}
            new_list.append(new_seed)

        # self.vis_seed_boxes(seed_all_frames)
        # set_trace()

        # if len(keep_mask_list) > 1:
        #     set_trace()

        return new_list

    def vis_seed_boxes(self, seed_all_frames):
        all_box = LiDARInstance3DBoxes.cat([s['gt_bboxes_3d'] for s in seed_all_frames])
        vis_bev_pc(None, all_box, name=f'vis_seed_boxes.png', dir='incremental')





    def get_seed_info(self, img_metas, offline_seed_info, cur_pose, device):
        if not self.test_cfg.get('reuse_results', False):
            return offline_seed_info

        assert self.test_cfg['sequential']

        assert len(self.previous_seed_info) == len(self.previous_pose)
        assert len(self.previous_seed_info) <= self.max_pre_frames

        cur_pose_inv = torch.linalg.inv(cur_pose)
        for seed, pre_pose in zip(self.previous_seed_info, self.previous_pose):
            boxes_in_curr = box_frame_transform_gpu(seed['gt_bboxes_3d'], pre_pose, None, cur_pose_inv=cur_pose_inv)
            seed['gt_bboxes_3d'] = boxes_in_curr

        # TODO: why fsd_inc_01 has performance drop using interval == 1
        calib_interval = self.test_cfg.get('calib_interval', -1)
        if calib_interval != -1 and (self.frame_counter + 1) % calib_interval == 0 and len(self.previous_seed_info) > 0:
            last_seed = copy.deepcopy(offline_seed_info[0])
            last_seed['gt_bboxes_3d'] = last_seed['gt_bboxes_3d'].to(device)
            last_seed['gt_labels_3d'] = torch.from_numpy(last_seed['gt_labels_3d']).to(device).float()
            last_seed['scores'] = torch.from_numpy(last_seed['scores']).to(device).float()
            self.previous_seed_info[-1] = last_seed

        if len(self.previous_seed_info) < self.max_pre_frames:
            return offline_seed_info
        else:
            # new_seed = copy.deepcopy(self.previous_seed_info)
            # new_seed.reverse()
            new_seed = [self.previous_seed_info[-i-1] for i in range(len(self.previous_seed_info))]
            return new_seed 
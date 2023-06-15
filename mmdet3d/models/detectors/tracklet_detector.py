import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result, LiDARTracklet

from mmdet.models import build_detector
from ..builder import build_backbone, build_head, build_neck

from mmseg.models import SEGMENTORS
from .. import builder
from mmdet3d.ops import scatter_v2, Voxelization, furthest_point_sample, get_inner_win_inds
from scipy.sparse.csgraph import connected_components
from mmdet.core import multi_apply
from .base import Base3DDetector
from mmdet3d.models.segmentors.base import Base3DSegmentor
# from mmdet3d.utils import vis_bev_pc
from ipdb import set_trace


@SEGMENTORS.register_module()
@DETECTORS.register_module()
class TrackletSegmentor(Base3DSegmentor):

    def __init__(self,
                 voxel_layer,
                 timestamp_encoder,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 segmentation_head,
                 decode_neck=None,
                 voxel_downsampling_size=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                 tanh_dims=None,
                 **extra_kwargs):
        super().__init__(init_cfg=init_cfg)

        self.voxel_layer = Voxelization(**voxel_layer)

        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.timestamp_encoder = TimestampEncoder(timestamp_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.backbone = build_backbone(backbone)
        if segmentation_head is not None:
            self.segmentation_head = build_head(segmentation_head)
            self.segmentation_head.train_cfg = train_cfg
            self.segmentation_head.test_cfg = test_cfg
            self.num_classes = segmentation_head['num_classes']
        self.decode_neck = build_neck(decode_neck)

        assert voxel_encoder['type'] == 'DynamicScatterVFE'


        self.print_info = {}
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cfg = train_cfg if train_cfg is not None else test_cfg
        self.save_list = []
        self.point_cloud_range = voxel_layer['point_cloud_range']
        self.voxel_size = voxel_layer['voxel_size']
        self.voxel_downsampling_size = voxel_downsampling_size
        self.tanh_dims = tanh_dims

    def encode_decode(self, ):
        return None

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        return NotImplementedError

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            if len(res) == 0:
                print('***********Attention: Got zero-point input***********')
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_feat(self, points, pts_frame_inds, img_metas):
        """Extract features from points."""
        batch_points, coors = self.voxelize(points)
        pts_frame_inds = torch.cat(pts_frame_inds, 0)
        coors = coors.long()

        pts_with_ts = self.timestamp_encoder(batch_points, pts_frame_inds)
        voxel_features, voxel_coors, voxel2point_inds = self.voxel_encoder(pts_with_ts, coors, return_inv=True)
        voxel_info = self.middle_encoder(voxel_features, voxel_coors)
        x = self.backbone(voxel_info)[0]
        padding = -1
        if 'shuffle_inds' not in voxel_info:
            voxel_feats_reorder = x['voxel_feats']
        else:
            # this branch only used in SST-based FSD 
            voxel_feats_reorder = self.reorder(x['voxel_feats'], voxel_info['shuffle_inds'], voxel_info['voxel_keep_inds'], padding) #'not consistent with voxel_coors any more'

        out = self.decode_neck(batch_points, coors, voxel_feats_reorder, voxel2point_inds, padding)

        return out, coors, batch_points, pts_frame_inds


    def reorder(self, data, shuffle_inds, keep_inds, padding=-1):
        '''
        Padding dropped voxel and reorder voxels.  voxel length and order will be consistent with the output of voxel_encoder.
        '''
        num_voxel_no_drop = len(shuffle_inds)
        data_dim = data.size(1)

        temp_data = padding * data.new_ones((num_voxel_no_drop, data_dim))
        out_data = padding * data.new_ones((num_voxel_no_drop, data_dim))

        temp_data[keep_inds] = data
        out_data[shuffle_inds] = temp_data

        return out_data

    def voxel_downsample(self, points_list):
        device = points_list[0].device
        out_points_list = []
        voxel_size = torch.tensor(self.voxel_downsampling_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)

        for points in points_list:
            coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
            out_points, new_coors = scatter_v2(points, coors, mode='avg', return_inv=False)
            out_points_list.append(out_points)
        return out_points_list

    def forward_train(self,
                      points,
                      pts_frame_inds,
                      img_metas,
                      ):
        if self.tanh_dims is not None:
            for p in points:
                p[:, self.tanh_dims] = torch.tanh(p[:, self.tanh_dims])
        elif points[0].size(1) in (4,5):
            # a hack way to scale the intensity and elongation in WOD
            points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]

        if self.voxel_downsampling_size is not None:
            points = self.voxel_downsample(points)

        # labels, vote_targets, vote_mask = self.segmentation_head.get_targets(points, gt_bboxes_3d, gt_labels_3d)

        neck_out, pts_coors, points, pts_frame_inds = self.extract_feat(points, pts_frame_inds, img_metas)

        feats = neck_out[0]
        valid_pts_mask = neck_out[1]
        points = points[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]
        pts_frame_inds = pts_frame_inds[valid_pts_mask]

        output_dict = dict(
            seg_points=points,
            seg_feats=feats,
            batch_idx=pts_coors[:, 0],
            pts_frame_inds=pts_frame_inds,
        )

        return output_dict


    def simple_test(self, points, pts_frame_inds, img_metas):

        # For now, simple_test process is the same with forward_train since the segmentor is actually a feature extractor
        return self.forward_train(points, pts_frame_inds, img_metas)





@DETECTORS.register_module()
class TrackletDetector(Base3DDetector):

    def __init__(self,
                 segmentor,
                 voxel_layer=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 neck=None,
                 bbox_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg,)
        if voxel_layer is not None:
            self.voxel_layer = Voxelization(**voxel_layer)
        if voxel_encoder is not None:
            self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        if middle_encoder is not None:
            self.middle_encoder = builder.build_middle_encoder(middle_encoder)

        self.segmentor = build_detector(segmentor)
        self.num_classes = roi_head['num_classes']

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cfg = self.train_cfg if self.train_cfg else self.test_cfg
        self.print_info = {}

        roi_head.update(train_cfg=train_cfg)
        roi_head.update(test_cfg=test_cfg)
        roi_head.pretrained = pretrained
        self.roi_head = builder.build_head(roi_head)

        # self.fake_linear = torch.nn.Linear(6, 6)

    def extract_feat(self,):
        """
        For abstract class instantiate
        """
        pass


    def forward_train(self,
                      points,
                      pts_frame_inds=None,
                      img_metas=None,
                      tracklet=None,
                      gt_tracklet_candidates=None
                      ):

        losses = {}
        # losses['loss_fake'] = self.fake_linear(points[0]).mean()

        # After PointsRangeFilter, there might be very few empty input.
        points = self.fake_points_for_empty_input(points)
        pts_frame_inds = self.fake_points_for_empty_input(pts_frame_inds)

        self.from_collate_format(tracklet, gt_tracklet_candidates)
        self.tracklet_to_device(tracklet, gt_tracklet_candidates, points[0].device)
        # return losses
        # vis_bev_pc(points[0], None, name=f'tracklet_pc_test.png', dir='tracklet', pc_range=100)

        seg_out_dict = self.segmentor.forward_train(points=points, pts_frame_inds=pts_frame_inds, img_metas=img_metas)

        data_dict = dict(
            points=seg_out_dict['seg_points'],
            pts_feats=seg_out_dict['seg_feats'],
            batch_idx=seg_out_dict['batch_idx'],
            pts_frame_inds=seg_out_dict['pts_frame_inds']
        )

        if self.cfg.get('pre_voxelization_size', None) is not None:
            data_dict = self.pre_voxelize_within_frame(data_dict)

        losses = self.roi_head.forward_train(
            pts_xyz=data_dict['points'][:, :3],
            pts_feats=data_dict['pts_feats'],
            pts_batch_idx=data_dict['batch_idx'],
            pts_frame_inds=data_dict['pts_frame_inds'],
            img_metas=img_metas,
            tracklet_list=tracklet,
            gt_candidates_list=gt_tracklet_candidates,
        )

        return losses

    def simple_test(self, points, img_metas, pts_frame_inds, tracklet, gt_tracklet_candidates=None, rescale=False):
        """Test function without augmentaiton."""

        points = self.fake_points_for_empty_input(points)
        pts_frame_inds = self.fake_points_for_empty_input(pts_frame_inds)

        self.from_collate_format(tracklet, gt_tracklet_candidates)
        self.tracklet_to_device(tracklet, gt_tracklet_candidates, points[0].device)

        seg_out_dict = self.segmentor.simple_test(points=points, pts_frame_inds=pts_frame_inds, img_metas=img_metas)

        data_dict = dict(
            points=seg_out_dict['seg_points'],
            pts_feats=seg_out_dict['seg_feats'],
            batch_idx=seg_out_dict['batch_idx'],
            pts_frame_inds=seg_out_dict['pts_frame_inds']
        )

        if self.cfg.get('pre_voxelization_size', None) is not None:
            data_dict = self.pre_voxelize_within_frame(data_dict)

        results = self.roi_head.simple_test(
            pts_xyz=data_dict['points'][:, :3],
            pts_feats=data_dict['pts_feats'],
            pts_batch_idx=data_dict['batch_idx'],
            pts_frame_inds=data_dict['pts_frame_inds'],
            img_metas=img_metas,
            tracklet_list=tracklet,
            gt_candidates_list=gt_tracklet_candidates,
        )

        return results

    def aug_test(self, points, img_metas, pts_frame_inds, tracklet, rescale=False):
        """Test function with augmentaiton."""
        assert len(points) == len(img_metas) == len(pts_frame_inds) == len(tracklet)
        aug_result_list = []
        for p, meta, inds, trk  in zip(points, img_metas, pts_frame_inds, tracklet):
            this_result = self.simple_test(p, meta, inds, trk)
            aug_result_list.append(this_result)
        bsz = len(points[0])
        num_augs = len(points)
        merged_result_list = []
        for i in range(bsz):
            aug_list_this_sample = []
            for k in range(num_augs):
                aug_list_this_sample.append(aug_result_list[k][i])
            this_merged_sample = LiDARTracklet.merge_augs(aug_list_this_sample, self.test_cfg['tta'], points[0][0].device)
            merged_result_list.append(this_merged_sample)

        return merged_result_list


    def pre_voxelize_within_frame(self, data_dict):
        batch_idx = data_dict['batch_idx']
        frame_inds = data_dict['pts_frame_inds']
        points = data_dict['points']
        pts_feats = data_dict['pts_feats']

        voxel_size = torch.tensor(self.cfg.pre_voxelization_size, device=batch_idx.device)
        pc_range = torch.tensor(self.cfg.point_cloud_range, device=points.device)
        coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], frame_inds[:, None], coors], dim=1)

        new_coors, unq_inv  = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)

        out_dict = {}
        for data_name in data_dict:
            data = data_dict[data_name]
            if data.dtype in (torch.float, torch.float16):
                voxelized_data, voxel_coors = scatter_v2(data, coors, mode='avg', return_inv=False, new_coors=new_coors, unq_inv=unq_inv)
                out_dict[data_name] = voxelized_data

        out_dict['batch_idx'] = voxel_coors[:, 0]
        out_dict['pts_frame_inds'] = voxel_coors[:, 1]
        return out_dict






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

    def tracklet_to_device(self, tracklets, candidates_list, device):
        for t in tracklets:
            t.to(device)

        if candidates_list is not None:
            for t_list in candidates_list:
                for t in t_list:
                    t.to(device)

    # def from_collate_format(self, tracklets, candidates_list=None):
    #     out_trks = [LiDARTracklet.from_collate_format(t) for t in tracklets]
    #     if candidates_list is not None:
    #         out_candidates_list = []
    #         for t_list in candidates_list:
    #             new_t_list = [LiDARTracklet.from_collate_format(t) for t in t_list]
    #             out_candidates_list.append(new_t_list)
    #         return out_trks, out_candidates_list
    #     return out_trks

    def from_collate_format(self, tracklets, candidates_list):
        for t in tracklets:
            t.from_collate_format()
        if candidates_list is not None:
            for t_list in candidates_list:
                for t in t_list:
                    t.from_collate_format()

    def fake_points_for_empty_input(self, points_list):
        new_points_list = []
        for p in points_list:
            if len(p) == 0:
                print('Empty input occurs!!!')
                if p.ndim == 1:
                    new_data = p.new_zeros((1,))
                else:
                    new_data = p.new_zeros((1, p.size(1)))
                new_points_list.append(new_data)
            else:
                new_points_list.append(p)
        return new_points_list

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Override the one in super to support batch inference
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))

        if self.test_cfg.get('tta', None) is not None:
            return self.aug_test(points, img_metas, **kwargs)
        else:
            img = [img] if img is None else img
            return self.simple_test(points, img_metas, **kwargs)


class TimestampEncoder(torch.nn.Module):
    ''' Generating cluster centers for each class and assign each point to cluster centers
    '''

    def __init__(
        self,
        strategy_cfg,
    ):
        super().__init__()
        self.strategy_cfg = strategy_cfg

    def forward(self, point, pts_frame_inds):
        # assert point.size(1) == 5, 'Rel time is removed in point loading'
        assert (point[:, -1] < 200).all()
        assert (pts_frame_inds >= 0).all()
        assert (pts_frame_inds <= 200).all(), 'Only holds on WOD'
        stra = self.strategy_cfg['strategy']
        return getattr(self, stra)(point, pts_frame_inds)

    def scalar(self, point, pts_frame_inds):
        n = self.strategy_cfg['normalizer']
        ts_embed = point[:, -1:] / n
        out = torch.cat([point, ts_embed], 1)
        return out
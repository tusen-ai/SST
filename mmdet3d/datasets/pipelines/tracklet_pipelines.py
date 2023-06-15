import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import os
from pdb import set_trace as st
import yaml
import torch
import torch.nn.functional as F
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.points import BasePoints, get_points_type
from collections import OrderedDict

import random
from scipy import signal


@PIPELINES.register_module()
class LoadTrackletPoints(object):

    def __init__(self, load_dim=5, use_dim=5, coord_type='LIDAR', max_points=-1, debug=False):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.coord_type = coord_type
        self.max_points = max_points
        self.debug = debug

    def __call__(self, input_dict):
        pts_filename = input_dict['pts_filename']
        ts_list = input_dict['tracklet'].ts_list
        ts0 = ts_list[0]
        rel_ts_list = [round((ts - ts0) / 1e5) for ts in ts_list]

        if self.debug:
            trk = input_dict['tracklet']
            points_list = [np.random.rand(100, 6).astype(np.float32) * 2 for i in range(len(trk))]
            for i, p in enumerate(points_list):
                p[:, :3] += trk.box_list[i].tensor[:, :3].float().numpy() 
        else:
            points_list = np.load(pts_filename, allow_pickle=True)
        if 'point_cloud_interval' in input_dict and input_dict['point_cloud_interval'] is not None:
            beg, end = input_dict['point_cloud_interval']
            points_list = points_list[beg:end]

        assert len(points_list) == len(input_dict['tracklet'])

        assert self.load_dim == points_list[0].shape[1]
        points_list = [p[:, :self.use_dim] for p in points_list]

        # points_list = [np.pad(p, ((0, 0), (0, 1)), mode='constant', constant_values=rel_ts) for p, rel_ts in zip(points_list, rel_ts_list)]
        points_list = [torch.from_numpy(p) for p in points_list]
        frame_inds_list = [torch.ones(len(p), dtype=torch.int) * i for i, p in enumerate(points_list)] # continuous frame_inds
        # points = np.concatenate(points_list, 0)
        # points_class = get_points_type(self.coord_type)
        # points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        if self.max_points > 0:
            points_list, frame_inds_list = self.points_downsample(points_list, frame_inds_list)
        input_dict['points'] = points_list
        input_dict['pts_frame_inds'] = frame_inds_list

        return input_dict

    def points_downsample(self, points_list, frame_inds_list):
        new_points_list, new_frame_inds_list = [], []
        for p, inds in zip(points_list, frame_inds_list):
            if len(p) > self.max_points:
                random_inds = torch.randperm(len(p))[:self.max_points]
                new_points_list.append(p[random_inds])
                new_frame_inds_list.append(inds[random_inds])
            else:
                new_points_list.append(p)
                new_frame_inds_list.append(inds)
        return new_points_list, new_frame_inds_list


@PIPELINES.register_module()
class LoadTrackletAnnotations(object):

    def __init__(self):
        return 

    def __call__(self, input_dict):
        input_dict['gt_tracklet_candidates'] = input_dict['ann_info']
        return input_dict

@PIPELINES.register_module()
class TrackletCutting(object):

    def __init__(
        self,
        min_length=5,
        ratio=0.5,
        max_cut_ratio=0.5,
        max_length=200,
        ):
        self.min_length = min_length
        self.ratio = ratio
        self.max_cut_ratio = max_cut_ratio
        self.max_length = max_length

    def __call__(self, input_dict):

        tracklet = input_dict['tracklet']
        if len(tracklet) < self.min_length or (np.random.rand() > self.ratio and len(tracklet) < self.max_length):
            return input_dict

        points_list = input_dict['points']
        pts_frame_inds = input_dict['pts_frame_inds']
        ts_list = tracklet.ts_list

        if len(tracklet) > self.max_length:
            cut_len = len(tracklet) - self.max_length
        else:
            cut_len = int(len(ts_list) * self.max_cut_ratio * np.random.rand())
        if cut_len < 1:
            return input_dict

        head = np.random.randint(0, cut_len)
        tail = cut_len - head
        cut_ts = ts_list[:head] + ts_list[-tail:]

        points_list = points_list[head: -tail]
        pts_frame_inds = pts_frame_inds[head: -tail]

        tracklet.remove(cut_ts)
        # candidates = input_dict['gt_tracklet_candidates']
        # for c in candidates:
        #     c.remove(cut_ts)
        assert len(tracklet) == len(points_list) == len(pts_frame_inds)

        input_dict['points'] = points_list
        input_dict['pts_frame_inds'] = pts_frame_inds

        return input_dict


@PIPELINES.register_module()
class TrackletPoseTransform(object):

    def __init__(self, concat=True, centering=False):
        self.concat = concat
        self.centering = centering


    def __call__(self, input_dict):
        points_list = input_dict['points']
        tracklet = input_dict['tracklet']
        pose_list = tracklet.pose_list


        assert not hasattr(tracklet, 'shared_pose') or tracklet.shared_pose is None
        assert len(points_list) == len(tracklet) == len(pose_list)

        center_pose = pose_list[len(pose_list)//2]

        tracklet.frame_transform(center_pose)

        if 'gt_tracklet_candidates' in input_dict:
            gt_candidates = input_dict['gt_tracklet_candidates']
            for trk in gt_candidates:
                trk.frame_transform(center_pose)

        tgt_pose_inv = torch.linalg.inv(center_pose)
        points_list = [
            torch.cat([self.points_frame_transform(p[:, :3], pose, None, tgt_pose_inv), p[:, 3:]], 1) 
            for pose, p in zip(pose_list, points_list)
        ]

        if self.centering:
            translation = -1 * tracklet.box_list[len(tracklet)//2].tensor[:, :3]
            for p in points_list:
                p[:, :3] += translation
            tracklet.translate(translation)
            if 'gt_tracklet_candidates' in input_dict:
                gt_candidates = input_dict['gt_tracklet_candidates']
                for trk in gt_candidates:
                    trk.translate(translation)
            tracklet.translation_factor = translation.numpy()


        input_dict['shared_pose'] = center_pose

        if self.concat:
            points = torch.cat(points_list, 0)
            points_class = get_points_type('LIDAR')
            points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
            input_dict['points'] = points
            input_dict['pts_frame_inds'] = torch.cat(input_dict['pts_frame_inds'])
        else:
            input_dict['points'] = points_list

        return input_dict

    def points_frame_transform(self, src_points, src_pose, tgt_pose, tgt_pose_inv=None):
        src_points_h = torch.nn.functional.pad(src_points, (0, 1), 'constant', 1)

        if tgt_pose_inv is None:
            world2tgt_pose = torch.inverse(tgt_pose)
        else:
            world2tgt_pose = tgt_pose_inv

        mm = world2tgt_pose @ src_pose
        tgt_points = (src_points_h @ mm.T)[:, :3]
        return tgt_points

@PIPELINES.register_module()
class TrackletGlobalRotScaleTrans(object):
    """Apply global rotation, scaling and translation to a 3D scene.
    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        input_dict['tracklet'].translate(trans_factor)
        if 'gt_tracklet_candidates' in input_dict: # not in the dict when TTA
            for trk in input_dict['gt_tracklet_candidates']:
                trk.translate(trans_factor)

    def _random_rot(self, input_dict):
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])
        input_dict['pcd_rot_angle'] = noise_rotation

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        # rotation = self.rot_range
        # noise_rotation = np.random.uniform(rotation[0], rotation[1])
        noise_rotation = input_dict['pcd_rot_angle']

        input_dict['tracklet'].rotate(noise_rotation)
        input_dict['tracklet'].rot_angle = noise_rotation
        if 'gt_tracklet_candidates' in input_dict: # not in the dict when TTA
            for trk in input_dict['gt_tracklet_candidates']:
                trk.rotate(noise_rotation)

        assert isinstance(input_dict['points'], BasePoints)
        input_dict['points'].rotate(-noise_rotation)

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        input_dict['points'].scale(scale)

        input_dict['tracklet'].scale(scale)
        if 'gt_tracklet_candidates' in input_dict: # not in the dict when TTA
            for trk in input_dict['gt_tracklet_candidates']:
                trk.scale(scale)


    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        assert input_dict['tracklet'].shared_pose is not None

        if 'gt_tracklet_candidates' in input_dict: # not in the dict when TTA
            if len(input_dict['gt_tracklet_candidates']) > 0:
                assert input_dict['gt_tracklet_candidates'][0].shared_pose is not None

        if 'pcd_rot_angle' not in input_dict:
            self._random_rot(input_dict)
        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str

@PIPELINES.register_module()
class TrackletRandomFlip(object):
    """Flip the points & bbox.
    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.
        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.
        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        input_dict['points'].flip(direction)
        input_dict['tracklet'].flip(direction)
        if 'gt_tracklet_candidates' in input_dict:
            for trk in input_dict['gt_tracklet_candidates']:
                trk.flip(direction)

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        # filp 2D image and its annotations

        if 'pcd_horizontal_flip' not in input_dict:
            flip_horizontal = True if np.random.rand() < self.flip_ratio_bev_horizontal else False
            input_dict['pcd_horizontal_flip'] = flip_horizontal

        if 'pcd_vertical_flip' not in input_dict:
            flip_vertical = True if np.random.rand() < self.flip_ratio_bev_vertical else False
            input_dict['pcd_vertical_flip'] = flip_vertical

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str

@PIPELINES.register_module()
class PointDecoration(object):

    def __init__(self, properties, concat=True):
        self.properties = properties
        self.concat = concat

    def __call__(self, input_dict):
        trk = input_dict['tracklet']
        assert trk.shared_pose is not None
        points_list = input_dict['points']

        for pro in self.properties:
            points_list = getattr(self, pro)(points_list, trk)

        if self.concat:
            points = torch.cat(points_list, 0)
            points_class = get_points_type('LIDAR')
            points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
            input_dict['points'] = points
            input_dict['pts_frame_inds'] = torch.cat(input_dict['pts_frame_inds'])
        else:
            input_dict['points'] = points_list

        return input_dict

    def yaw(self, points_list, trk):
        assert len(points_list) == len(trk)
        box_list = trk.box_list
        yaw_list = [b.tensor[0, 6].item() for b in box_list]
        points_list = [F.pad(p, (0, 1), 'constant', yaw / 3.1415) for p, yaw in zip(points_list, yaw_list)]
        return points_list

    def size(self, points_list, trk):
        assert len(points_list) == len(trk)
        box_list = trk.box_list
        size_list = [b.tensor[:, 3:6] / 10 for b in box_list]
        points_list = [torch.cat([p, s.expand(len(p), -1)], 1) for p, s in zip(points_list, size_list)]
        return points_list

    def score(self, points_list, trk):
        assert len(points_list) == len(trk)
        score_list = trk.score_list
        points_list = [F.pad(p, (0, 1), 'constant', s) for p, s in zip(points_list, score_list)]
        return points_list

    def center_offset(self, points_list, trk):
        new_list = []
        for i, p in enumerate(points_list):
            box_center = trk.box_list[i].tensor[:, :3]
            new_p = torch.cat([p, (p[:, :3] - box_center)/5], 1) 
            new_list.append(new_p)
        return new_list

    def length(self, points_list, trk):
        length = len(trk)
        # score_list = trk.score_list
        points_list = [F.pad(p, (0, 1), 'constant', length/100) for p in points_list]
        return points_list


@PIPELINES.register_module()
class FrameDropout(object):

    def __init__(self, drop_ratio=0.1):
        self.drop_ratio = drop_ratio

    def __call__(self, input_dict):
        tracklet = input_dict['tracklet']
        points_list = input_dict['points']
        frame_inds_list = input_dict['pts_frame_inds']

        drop_ts, keep_idx = tracklet.random_frame_drop(self.drop_ratio)

        points_list = [points_list[i] for i in keep_idx]
        frame_inds_list = [frame_inds_list[i] for i in keep_idx]

        input_dict['points'] = points_list
        input_dict['pts_frame_inds'] = frame_inds_list

        assert len(points_list) == len(tracklet)

        # if 'gt_tracklet_candidates' in input_dict:
        #     gt_candidates = input_dict['gt_tracklet_candidates']
        #     for trk in gt_candidates:
        #         _ = trk.remove(drop_ts)

        return input_dict

@PIPELINES.register_module()
class TrackletNoise(object):

    def __init__(
        self,
        center_noise_cfg=None,
        size_noise_cfg=None,
        yaw_noise_cfg=None,
        ):
        self.c_cfg = center_noise_cfg
        self.s_cfg = size_noise_cfg
        self.y_cfg = yaw_noise_cfg

    def __call__(self, input_dict):
        tracklet = input_dict['tracklet']

        if self.c_cfg is not None:
            tracklet.add_center_noise(self.c_cfg['max_noise'], self.c_cfg['consistent'])

        if self.s_cfg is not None:
            tracklet.add_size_noise(self.s_cfg['max_noise'], self.s_cfg['consistent'])

        if self.y_cfg is not None:
            tracklet.add_yaw_noise(self.y_cfg['max_noise'], self.y_cfg['consistent'])

        return input_dict


@PIPELINES.register_module()
class TrackletScaling(object):

    def __init__(
        self,
        max_step=0.1,
        ratio=0.2,
        ignore_yaw_thresh=0.78539,
        median_filter_size=3,
        reverse_heading=True,
        ):
        self.max_step = max_step
        self.ratio = ratio
        self.ignore_yaw_thresh = ignore_yaw_thresh
        self.median_filter_size = median_filter_size
        self.reverse_heading = reverse_heading

    def __call__(self, input_dict):
        if np.random.rand() > self.ratio or len(input_dict['tracklet']) < self.median_filter_size:
            return input_dict

        movements = self.compute_movements(input_dict)
        if movements is None:
            return input_dict

        tracklet = input_dict['tracklet']
        points_list = input_dict['points']
        assert len(movements) == len(tracklet) == len(movements)

        for i, m in enumerate(movements):
            box = tracklet.box_list[i]
            pts = points_list[i]
            box.translate(m.squeeze(0))
            pts[:, :3] += m

        ts_list = tracklet.ts_list
        if 'gt_tracklet_candidates' in input_dict:
            candidates = input_dict['gt_tracklet_candidates']
            for c in candidates:
                c.translate_by_ts(ts_list, movements)

        # from ipdb import set_trace
        # set_trace()


        return input_dict

    def compute_movements(self, input_dict):
        pi = 3.1415926
        tracklet = input_dict['tracklet']
        beg_yaw = tracklet.box_list[0].tensor[0, 6].item()
        end_yaw = tracklet.box_list[-1].tensor[0, 6].item()
        yaw_diff = beg_yaw - end_yaw
        if yaw_diff < - pi:
            yaw_diff += 2 * pi
        if yaw_diff > pi:
            yaw_diff -= 2 * pi
        yaw_diff = abs(yaw_diff)
        if yaw_diff > self.ignore_yaw_thresh:
            return None

        concated_boxes = tracklet.concated_boxes()
        yaws = concated_boxes.tensor[:, 6].numpy()
        smooth_yaws = torch.from_numpy(signal.medfilt(yaws, self.median_filter_size))

        sin = torch.sin(smooth_yaws)
        cos = torch.cos(smooth_yaws)
        heading_vectors = torch.stack([sin, cos], 1)
        heading_vectors = F.pad(heading_vectors, (0, 1), 'constant', 0)
        if self.reverse_heading:
            heading_vectors = -1 * heading_vectors

        # heading_vectors = concated_boxes.heading_unit_vector()[:, :3] #[N, 3]
        ######
        mean_length = concated_boxes.tensor[:, 4].mean()
        single_movements = heading_vectors * mean_length * self.max_step * torch.rand(1)
        movements = torch.cumsum(single_movements, 0)
        max_movement = movements[-1, :]
        movements -= max_movement[None, :]/2
        assert (movements[:, -1] == 0).all()
        movements = torch.split(movements, 1, 0)
        return movements
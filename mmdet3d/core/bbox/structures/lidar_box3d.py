import numpy as np
import torch

from mmdet3d.core.points import BasePoints
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
from mmdet3d.ops.iou3d import iou3d_cuda
from .base_box3d import BaseInstance3DBoxes
from .utils import limit_period, rotation_3d_in_axis, xywhr2xyxyr
try:
    from torchex import boxes_overlap_1to1
except ImportError:
    boxes_overlap_1to1 = None


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                            up z    x front (yaw=-0.5*pi)
                               ^   ^
                               |  /
                               | /
      (yaw=-pi) left y <------ 0 -------- (yaw=0)

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the negative direction of y axis, and decreases from
    the negative direction of y to the positive direction of x.

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / oriign    | /
            left y<-------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)
        """
        # TODO: rotation_3d_in_axis function do not support
        #  empty tensor currently.
        assert len(self.tensor) != 0
        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=2)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    @property
    def canonical_corners(self):
        assert len(self.tensor) != 0
        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
        return corners

    @property
    def heading_unit_vector(self):
        yaw = self.tensor[:, 6]
        sin = torch.sin(yaw)
        cos = torch.cos(yaw)
        vector = torch.stack([sin, cos, torch.zeros_like(sin)], -1)
        return vector

    @property
    def bev(self):
        """torch.Tensor: 2D BEV box of each box with rotation
        in XYWHR format."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def nearest_bev(self):
        """torch.Tensor: A tensor of 2D BEV box of each box
        without rotation."""
        # Obtain BEV boxes with rotation in XYWHR format
        bev_rotated_boxes = self.bev
        # convert the rotation to a valid range
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))

        # find the center of boxes
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(conditions, bev_rotated_boxes[:,
                                                                [0, 1, 3, 2]],
                                  bev_rotated_boxes[:, :4])

        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or \
        rotation matrix.

        Args:
            angles (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns \
                None, otherwise it returns the rotated points and the \
                rotation matrix ``rot_mat_T``.
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            rot_sin = torch.sin(angle)
            rot_cos = torch.cos(angle)
            rot_mat_T = self.tensor.new_tensor([[rot_cos, -rot_sin, 0],
                                                [rot_sin, rot_cos, 0],
                                                [0, 0, 1]])
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[1, 0]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)

        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T
        self.tensor[:, 6] += angle

        if self.tensor.shape[1] == 9:
            # rotate velo vector
            self.tensor[:, 7:9] = self.tensor[:, 7:9] @ rot_mat_T[:2, :2]

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                # clockwise
                points.rotate(-angle)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(self, bev_direction='horizontal', points=None):
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, None):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == 'vertical':
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 1] = -points[:, 1]
                elif bev_direction == 'vertical':
                    points[:, 0] = -points[:, 0]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): the range of box
                (x_min, y_min, x_max, y_max)

        Note:
            The original implementation of SECOND checks whether boxes in
            a range by checking whether the points are in a convex
            polygon, we reduce the burden for simpler cases.

        Returns:
            torch.Tensor: Whether each box is inside the reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > box_range[0])
                          & (self.tensor[:, 1] > box_range[1])
                          & (self.tensor[:, 0] < box_range[2])
                          & (self.tensor[:, 1] < box_range[3]))
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`BoxMode`): the target Box mode
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from ``src`` coordinates to ``dst`` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: \
                The converted box of the same type in the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode
        return Box3DMode.convert(
            box=self, src=Box3DMode.LIDAR, dst=dst, rt_mat=rt_mat)

    def enlarged_box(self, extra_width):
        """Enlarge the length, width and height boxes.

        Args:
            extra_width (float | torch.Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        if extra_width < 0:
            invalid_mask = (enlarged_boxes[:, 3:6] <= 0).any(1)
            enlarged_boxes[invalid_mask] = self.tensor[invalid_mask]
        return self.new_box(enlarged_boxes)

    def classwise_enlarged_box(self, extra_width_dict, labels):
        """Enlarge the length, width and height boxes.

        Args:
            extra_width (float | torch.Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        unq_labels = torch.unique(labels).tolist()
        extra_width = self.tensor.new_ones(len(self.tensor))
        for l in unq_labels:
            this_mask = labels == l
            extra_width[this_mask] = extra_width_dict[l]
        extra_width = extra_width[:, None]

        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width.squeeze()
        return self.new_box(enlarged_boxes)
    
    def noisy_box(self, center_noise, dim_noise, yaw_noise):
        num = len(self.tensor)
        device = self.tensor.device
        dtype = self.tensor.dtype

        if isinstance(center_noise, (list, tuple)):
            center_noise = torch.tensor(center_noise, dtype=rois.dtype, device=rois.device)[None, :]

        if isinstance(dim_noise, (list, tuple)):
            dim_noise = torch.tensor(dim_noise, dtype=rois.dtype, device=rois.device)[None, :]
        
        dims = self.tensor[:, 3:6]

        cen_noise = (torch.rand((num, 3), dtype=dtype, device=device) - 0.5) * 2 * center_noise * dims
        dim_noise = (torch.rand((num, 3), dtype=dtype, device=device) - 0.5) * 2 * dim_noise + 1
        yaw_noise = (torch.rand((num,  ), dtype=dtype, device=device) - 0.5) * 2 * yaw_noise
        noisy_boxes = self.tensor.clone()
        noisy_boxes[:, :3] += cen_noise
        noisy_boxes[:, 3:6] *= dim_noise
        noisy_boxes[:, 6] += yaw_noise
        return self.new_box(noisy_boxes)

    def enlarged_box_hw(self, extra_width):
        """Enlarge the length, width and height boxes.

        Args:
            extra_width (float | torch.Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:5] += extra_width * 2
        # bottom center z minus extra_width
        if extra_width < 0:
            invalid_mask = (enlarged_boxes[:, 3:5] <= 0).any(1)
            enlarged_boxes[invalid_mask] = self.tensor[invalid_mask]
        return self.new_box(enlarged_boxes)

    def points_in_boxes(self, points):
        """Find the box which the points are in.

        Args:
            points (torch.Tensor): Points in shape (N, 3).

        Returns:
            torch.Tensor: The index of box where each point are in.
        """
        box_idx = points_in_boxes_gpu(
            points.unsqueeze(0),
            self.tensor[:, :7].unsqueeze(0).to(points.device)).squeeze(0)
        return box_idx

    def move(self, t=0.1):
        """ move boxes along the velocity

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        assert not self.moved
        velo = self.tensor[:, [7, 8]]
        moved_boxes = self.tensor.clone()
        moved_boxes[:, :2] += velo * t
        # bottom center z minus extra_width
        return self.new_box(moved_boxes, moved=True)

    @classmethod
    def aligned_height_overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate height overlaps of two boxes.
        Note:
            This function calculates the height overlaps between boxes1 and
            boxes2,  boxes1 and boxes2 should be in the same type.
        Args:
            boxes1 (:obj:`BaseInstanceBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstanceBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.
        Returns:
            torch.Tensor: Calculated iou of boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

        boxes1_top_height = boxes1.top_height.view(-1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1)
        boxes2_top_height = boxes2.top_height.view(-1)
        boxes2_bottom_height = boxes2.bottom_height.view(-1)

        heighest_of_bottom = torch.max(boxes1_bottom_height, boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    @classmethod
    def aligned_iou_3d(cls, boxes1, boxes2, mode='iou'):
        """Calculate 3D overlaps of two boxes.
        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.
        Args:
            boxes1 (:obj:`BaseInstanceBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstanceBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.
        Returns:
            torch.Tensor: Calculated iou of boxes' heights.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'


        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        # height overlap
        overlaps_h = cls.aligned_height_overlaps(boxes1, boxes2)

        # obtain BEV boxes in XYXYR format
        boxes1_bev = xywhr2xyxyr(boxes1.bev)
        boxes2_bev = xywhr2xyxyr(boxes2.bev)

        overlaps_bev = boxes_overlap_1to1(boxes1_bev, boxes2_bev)

        # 3d overlaps
        overlaps_3d = overlaps_bev * overlaps_h

        volume1 = boxes1.volume.view(-1)
        volume2 = boxes2.volume.view(-1)

        if mode == 'iou':
            # the clamp func is used to avoid division of 0
            iou3d = overlaps_3d / torch.clamp(
                volume1 + volume2 - overlaps_3d, min=1e-8)
        else:
            iou3d = overlaps_3d / torch.clamp(volume1, min=1e-8)

        return iou3d

    @classmethod
    def overlaps_bev(cls, boxes1, boxes2, mode='iou'):
        """Calculate 3D overlaps of two boxes.
        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.
        Args:
            boxes1 (:obj:`BaseInstanceBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstanceBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.
        Returns:
            torch.Tensor: Calculated iou of boxes' heights.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

        assert mode in ['iou', 'iof']

        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        # height overlap

        # obtain BEV boxes in XYXYR format
        boxes1_bev = xywhr2xyxyr(boxes1.bev)
        boxes2_bev = xywhr2xyxyr(boxes2.bev)

        # bev overlap
        overlaps_bev = boxes1_bev.new_zeros(
            (boxes1_bev.shape[0], boxes2_bev.shape[0])).cuda()  # (N, M)
        iou3d_cuda.boxes_overlap_bev_gpu(boxes1_bev.contiguous().cuda(),
                                         boxes2_bev.contiguous().cuda(),
                                         overlaps_bev)

        # 3d overlaps
        overlaps_bev = overlaps_bev.to(boxes1.device) 

        area1 = boxes1.area.view(-1, 1)
        area2 = boxes2.area.view(1, -1)

        if mode == 'iou':
            # the clamp func is used to avoid division of 0
            iou_bev = overlaps_bev / torch.clamp(
                area1 + area2 - overlaps_bev, min=1e-8)
        else:
            iou_bev = overlaps_bev / torch.clamp(area1, min=1e-8)

        return iou_bev

    @classmethod
    def aligned_iou_bev(cls, boxes1, boxes2, mode='iou'):
        """Calculate 3D overlaps of two boxes.
        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.
        Args:
            boxes1 (:obj:`BaseInstanceBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstanceBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.
        Returns:
            torch.Tensor: Calculated iou of boxes' heights.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'


        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        # height overlap
        # overlaps_h = cls.aligned_height_overlaps(boxes1, boxes2)

        # obtain BEV boxes in XYXYR format
        boxes1_bev = xywhr2xyxyr(boxes1.bev)
        boxes2_bev = xywhr2xyxyr(boxes2.bev)

        overlaps_bev = boxes_overlap_1to1(boxes1_bev, boxes2_bev)

        # 3d overlaps
        overlaps_3d = overlaps_bev

        volume1 = boxes1.area.view(-1)
        volume2 = boxes2.area.view(-1)

        if mode == 'iou':
            # the clamp func is used to avoid division of 0
            iou3d = overlaps_3d / torch.clamp(
                volume1 + volume2 - overlaps_3d, min=1e-8)
        else:
            iou3d = overlaps_3d / torch.clamp(volume1, min=1e-8)

        return iou3d
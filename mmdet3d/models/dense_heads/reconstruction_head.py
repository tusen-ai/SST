import numpy as np
import torch
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss, binary_cross_entropy_with_logits
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import build_norm_layer
from torch import nn as nn

from mmdet3d.core import (PseudoSampler, box3d_multiclass_nms, limit_period,
                          xywhr2xyxyr, box3d_multiclass_wnms)
from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from mmdet.models import HEADS
from ..builder import build_loss
from .train_mixins import AnchorTrainMixin

from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes

from ipdb import set_trace
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS
from mmcv.utils import build_from_cfg


@HEADS.register_module()
class ReconstructionHead(BaseModule):
    """Anchor head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
    """

    def __init__(self,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 feat_channels=256,
                 num_reg_points=10,
                 only_masked=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_reg_points = num_reg_points
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        # build loss function
        self.only_masked = only_masked
        self.fp16_enabled = False

        self._init_layers()

        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv1d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_occupied', std=0.01, bias_prob=0.01))

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.conv_occupied = nn.Conv1d(self.feat_channels, 1, 1)
        self.conv_num_points = nn.Conv1d(self.feat_channels, 1, 1)
        # self.conv_reg = nn.Conv1d(self.feat_channels, self.num_reg_points * 3, 1)

    def _apply_1dconv(self, conv, x):
        x = x.unsqueeze(0).transpose(1, 2)
        x = conv(x).transpose(1, 2).squeeze(0)
        return x

    def forward(self, x):
        """Forward pass.

        Args:r
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        voxel_info, voxel_info_decoder, voxel_info_encoder = x
        predictions = voxel_info_decoder["output"]  # [N, C]
        gt_dict = voxel_info_decoder["gt_dict"]

        masked_predictions = predictions[voxel_info_decoder["dec2masked_idx"]]
        unmasked_predictions = predictions[voxel_info_decoder["dec2unmasked_idx"]]

        # TODO: Do occupied loss
        pred_occupied = self._apply_1dconv(self.conv_occupied, predictions).view(-1)

        # Predict number of points loss
        pred_num_points_masked = self._apply_1dconv(self.conv_num_points, masked_predictions).view(-1)
        pred_num_points_unmasked = self._apply_1dconv(self.conv_num_points, unmasked_predictions).view(-1)


        gt_num_points = gt_dict["num_points_per_voxel"]
        gt_num_points_masked = gt_num_points[voxel_info_decoder["masked_idx"]]
        gt_num_points_unmasked = gt_num_points[voxel_info_decoder["unmasked_idx"]]

        pred_dict = {}

        pred_dict["pred_occupied"] = pred_occupied
        pred_dict["gt_occupied"] = torch.ones_like(
            pred_occupied, dtype=pred_occupied.dtype, device=pred_occupied.device)

        pred_dict["pred_num_points_masked"] = pred_num_points_masked
        pred_dict["pred_num_points_unmasked"] = pred_num_points_unmasked
        pred_dict["gt_num_points_masked"] = gt_num_points_masked
        pred_dict["gt_num_points_unmasked"] = gt_num_points_unmasked

        return (pred_dict,)

    def loss(self,
             pred_dict,
             gt_bboxes,
             gt_labels,
             input_metas,
             gt_bboxes_ignore=None,
             ):
        """Calculate losses.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Gt bboxes
                of each sample.
            gt_labels (list[torch.Tensor]): Gt labels of each sample.
            input_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and \
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification \
                    losses.
        """
        pred_occupied = pred_dict["pred_occupied"]
        gt_occupied = pred_dict["gt_occupied"]
        loss_occupied = binary_cross_entropy_with_logits(pred_occupied, gt_occupied)

        pred_num_points_masked = pred_dict["pred_num_points_masked"]
        pred_num_points_unmasked = pred_dict["pred_num_points_unmasked"]
        gt_num_points_masked = pred_dict["gt_num_points_masked"]
        gt_num_points_unmasked = pred_dict["gt_num_points_unmasked"]
        loss_num_points_masked = smooth_l1_loss(pred_num_points_masked, gt_num_points_masked.float(), reduction="mean")
        loss_num_points_unmasked = smooth_l1_loss(pred_num_points_unmasked, gt_num_points_unmasked.float(), reduction="mean")

        return dict(
            loss_occupied=loss_occupied,
            loss_num_points_masked=loss_num_points_masked,
            loss_num_points_unmasked=loss_num_points_unmasked)


from mmcv.cnn.bricks import ConvModule
import torch
from torch import nn as nn

from mmdet3d.ops import PointFPModule
from mmdet.models import HEADS
from .decode_head import Base3DDecodeHead
from ipdb import set_trace
from mmcv.runner import auto_fp16, force_fp32, BaseModule
from mmcv.cnn import build_norm_layer, normal_init
from mmseg.models.builder import build_loss
from mmdet.models.builder import build_loss as build_det_loss

from mmdet3d.ops import build_mlp, scatter_v2
from torch.utils.checkpoint import checkpoint


@HEADS.register_module()
class SimpleSegHead(Base3DDecodeHead):

    def __init__(self,
                 in_channel,
                 num_classes,
                 hidden_dims=[],
                 dropout_ratio=0.5,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='naiveSyncBN1d'),
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_aux=None,
                 ignore_index=255,
                 logit_scale=1,
                 init_cfg=None):
        end_channel = hidden_dims[-1] if len(hidden_dims) > 0 else in_channel
        super(SimpleSegHead, self).__init__(
                 end_channel,
                 num_classes,
                 dropout_ratio,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 loss_decode,
                 ignore_index,
                 init_cfg
        )

        self.pre_seg_conv = None
        if len(hidden_dims) > 0:
            layer_list = []
            last_channel = in_channel
            for c in hidden_dims:
                norm_layer = build_norm_layer(norm_cfg, c)[1]
                layer_list.append(
                    nn.Sequential(
                        nn.Linear(last_channel, c, bias=False),
                        norm_layer,
                        nn.ReLU(inplace=True)
                    )
                )
                last_channel = c
            self.pre_seg_conv = nn.Sequential(*layer_list)

        self.logit_scale = logit_scale
        self.conv_seg = nn.Linear(end_channel, num_classes)
        self.fp16_enabled = False

        if loss_aux is not None:
            self.loss_aux = build_loss(loss_aux)
        else:
            self.loss_aux = None
        if loss_decode['type'] == 'FocalLoss':
            self.loss_decode = build_det_loss(loss_decode) # mmdet has a better focal loss supporting single class
            self.binary_seg = True
        else:
            self.binary_seg = False


    @auto_fp16(apply_to=('voxel_feat',))
    def forward(self, voxel_feat):
        """Forward pass.

        """

        output = voxel_feat
        if self.pre_seg_conv is not None:
            output = self.pre_seg_conv(voxel_feat)
        output = self.cls_seg(output)

        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted per-point segmentation logits \
                of shape [B, num_classes, N].
            seg_label (torch.Tensor): Ground-truth segmentation label of \
                shape [B, N].
        """
        seg_logit = seg_logit * self.logit_scale
        loss = dict()
        if self.binary_seg:
            loss['loss_sem_seg'] = self.loss_decode(seg_logit, seg_label)
        else:
            loss['loss_sem_seg'] = self.loss_decode(
                seg_logit, seg_label, ignore_index=self.ignore_index)
        if self.loss_aux is not None:
            loss['loss_aux'] = self.loss_aux(seg_logit, seg_label)

        is_right = seg_logit.argmax(1) == seg_label
        loss['acc'] = is_right.sum().float() / len(is_right)

        return loss

    def forward_train(self, inputs, img_metas, pts_semantic_mask, train_cfg, return_logit=False):

        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, pts_semantic_mask)
        if return_logit:
            return losses, dict(seg_logit=seg_logits)
        else:
            return losses

@HEADS.register_module()
class VarianceSegHead(Base3DDecodeHead):

    def __init__(self,
                 in_channel,
                 num_classes,
                 hidden_dims=[],
                 dropout_ratio=0.5,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='naiveSyncBN1d'),
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_var=dict(
                     type='SmoothL1Loss',
                     beta=1.0/9.0,
                     loss_weight=1.0),
                 loss_aux=None,
                 ignore_index=255,
                 logit_scale=1,
                 init_cfg=None):
        end_channel = hidden_dims[-1] if len(hidden_dims) > 0 else in_channel
        super(VarianceSegHead, self).__init__(
                 end_channel,
                 num_classes,
                 dropout_ratio,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 loss_decode,
                 ignore_index,
                 init_cfg
        )

        self.pre_seg_conv = None
        if len(hidden_dims) > 0:
            layer_list = []
            last_channel = in_channel
            for c in hidden_dims:
                norm_layer = build_norm_layer(norm_cfg, c)[1]
                layer_list.append(
                    nn.Sequential(
                        nn.Linear(last_channel, c, bias=False),
                        norm_layer,
                        nn.ReLU(inplace=True)
                    )
                )
                last_channel = c
            self.pre_seg_conv = nn.Sequential(*layer_list)

        self.logit_scale = logit_scale
        self.conv_seg = nn.Linear(end_channel, num_classes)
        self.conv_var = nn.Linear(end_channel, 1)
        self.fp16_enabled = False
        self.loss_var = build_det_loss(loss_var)

        if loss_aux is not None:
            self.loss_aux = build_loss(loss_aux)
        else:
            self.loss_aux = None

    @auto_fp16(apply_to=('voxel_feat',))
    def forward(self, voxel_feat):
        """Forward pass.

        """

        output = voxel_feat
        if self.pre_seg_conv is not None:
            output = self.pre_seg_conv(voxel_feat)
        logit = self.cls_seg(output)
        var = self.conv_var(output)
        var = var.reshape(-1)
        result = dict(logit=logit, pred_loss=var)

        return result

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, pred_var, seg_label):
        """Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted per-point segmentation logits \
                of shape [B, num_classes, N].
            seg_label (torch.Tensor): Ground-truth segmentation label of \
                shape [B, N].
        """
        seg_logit = seg_logit * self.logit_scale
        loss = dict()
        loss['loss_sem_seg'] = self.loss_decode(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        if self.loss_aux is not None:
            loss['loss_aux'] = self.loss_aux(seg_logit, seg_label)

        no_grad_loss = self.no_grad_ce_loss(seg_logit, seg_label)
        loss['loss_var'] = self.loss_var(pred_var, no_grad_loss)
        loss['no_grad_ce'] = no_grad_loss
            # loss_bbox = self.loss_bbox(
            #     pos_bbox_pred,
            #     pos_bbox_targets,
            #     pos_bbox_weights,
            #     avg_factor=num_total_samples)

        return loss

    def forward_train(self, inputs, img_metas, pts_semantic_mask, train_cfg, return_logit=False):

        output = self.forward(inputs)
        seg_logits = output['logit']
        pred_var = output['pred_loss']
        losses = self.losses(seg_logits, pred_var, pts_semantic_mask)
        if return_logit:
            return losses, dict(seg_logit=seg_logits)
        else:
            return losses
    
    @torch.no_grad()
    def no_grad_ce_loss(self, logit, label):
        ce_loss = nn.functional.cross_entropy(logit, label, reduction='none')
        return ce_loss

@HEADS.register_module()
class AssertionSegHead(VarianceSegHead):

    def __init__(self,
                 in_channel,
                 num_classes,
                 hidden_dims=[],
                 dropout_ratio=0.5,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='naiveSyncBN1d'),
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_var=dict(
                     type='FocalLoss',
                     alpha=0.25,
                     gamma=2,
                     loss_weight=1.0),
                 loss_aux=None,
                 ignore_index=255,
                 logit_scale=1,
                 init_cfg=None):
        super(AssertionSegHead, self).__init__(
                 in_channel,
                 num_classes,
                 hidden_dims,
                 dropout_ratio,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 loss_decode,
                 loss_var,
                 loss_aux,
                 ignore_index,
                 logit_scale,
                 init_cfg,
        )

    @auto_fp16(apply_to=('voxel_feat',))
    def forward(self, voxel_feat):
        """Forward pass.

        """

        output = voxel_feat
        if self.pre_seg_conv is not None:
            output = self.pre_seg_conv(voxel_feat)
        logit = self.cls_seg(output)
        var = self.conv_var(output)
        result = dict(logit=logit, assert_logit=var)

        return result

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, assert_logit, seg_label):
        """Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted per-point segmentation logits \
                of shape [B, num_classes, N].
            seg_label (torch.Tensor): Ground-truth segmentation label of \
                shape [B, N].
        """
        seg_logit = seg_logit * self.logit_scale
        loss = dict()
        loss['loss_sem_seg'] = self.loss_decode(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        if self.loss_aux is not None:
            loss['loss_aux'] = self.loss_aux(seg_logit, seg_label)

        is_right = seg_logit.argmax(1) == seg_label
        assert_label = 1 - is_right.long() # 0 is right, 1 is wrong (max_label usually equal to feat_dim, so max_label indicates negative)
        loss['loss_assert'] = self.loss_var(assert_logit, assert_label)

        # log
        loss['acc'] = is_right.sum().float() / len(is_right)
        assert_false = assert_logit.squeeze(1).sigmoid() < 0.5
        tp = (assert_false & (is_right == 0)).sum().float()
        total_assert_false = assert_false.sum().float()
        total_false = (is_right == 0).sum().float()
        loss['assert_false_acc'] =  tp / (total_assert_false + 1e-5)
        loss['assert_false_recall'] =  tp / (total_false + 1e-5)

        return loss

    def forward_train(self, inputs, img_metas, pts_semantic_mask, train_cfg, return_logit=False):

        output = self.forward(inputs)
        seg_logits = output['logit']
        assert_logit = output['assert_logit']
        losses = self.losses(seg_logits, assert_logit, pts_semantic_mask)
        if return_logit:
            return losses, dict(seg_logit=seg_logits, assert_logit=assert_logit)
        else:
            return losses

@HEADS.register_module()
class VoteSegHead(Base3DDecodeHead):

    def __init__(self,
                 in_channel,
                 num_classes,
                 hidden_dims=[],
                 dropout_ratio=0.5,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='naiveSyncBN1d'),
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_vote=dict(
                     type='L1Loss',
                 ),
                 loss_aux=None,
                 ignore_index=255,
                 logit_scale=1,
                 checkpointing=False,
                 init_bias=None,
                 init_cfg=None):
        end_channel = hidden_dims[-1] if len(hidden_dims) > 0 else in_channel
        super(VoteSegHead, self).__init__(
                 end_channel,
                 num_classes,
                 dropout_ratio,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 loss_decode,
                 ignore_index,
                 init_cfg
        )

        self.pre_seg_conv = None
        if len(hidden_dims) > 0:
            self.pre_seg_conv = build_mlp(in_channel, hidden_dims, norm_cfg, act=act_cfg['type'])

        self.use_sigmoid = loss_decode.get('use_sigmoid', False)
        self.bg_label = self.num_classes
        if not self.use_sigmoid:
            self.num_classes += 1


        self.logit_scale = logit_scale
        self.conv_seg = nn.Linear(end_channel, self.num_classes)
        self.voting = nn.Linear(end_channel, self.num_classes * 3)
        self.fp16_enabled = False
        self.checkpointing = checkpointing
        self.init_bias = init_bias

        if loss_aux is not None:
            self.loss_aux = build_loss(loss_aux)
        else:
            self.loss_aux = None
        if loss_decode['type'] == 'FocalLoss':
            self.loss_decode = build_det_loss(loss_decode) # mmdet has a better focal loss supporting single class
        
        self.loss_vote = build_det_loss(loss_vote)

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        if self.init_bias is not None:
            self.conv_seg.bias.data.fill_(self.init_bias)
            print(f'Segmentation Head bias is initialized to {self.init_bias}')
        else:
            normal_init(self.conv_seg, mean=0, std=0.01)


    @auto_fp16(apply_to=('voxel_feat',))
    def forward(self, voxel_feat):
        """Forward pass.

        """

        output = voxel_feat
        if self.pre_seg_conv is not None:
            if self.checkpointing:
                output = checkpoint(self.pre_seg_conv, voxel_feat)
            else:
                output = self.pre_seg_conv(voxel_feat)
        logits = self.cls_seg(output)
        vote_preds = self.voting(output)

        return logits, vote_preds

    @force_fp32(apply_to=('seg_logit', 'vote_preds'))
    def losses(self, seg_logit, vote_preds, seg_label, vote_targets, vote_mask):
        """Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted per-point segmentation logits \
                of shape [B, num_classes, N].
            seg_label (torch.Tensor): Ground-truth segmentation label of \
                shape [B, N].
        """
        seg_logit = seg_logit * self.logit_scale
        loss = dict()
        loss['loss_sem_seg'] = self.loss_decode(seg_logit, seg_label)
        if self.loss_aux is not None:
            loss['loss_aux'] = self.loss_aux(seg_logit, seg_label)

        vote_preds = vote_preds.reshape(-1, self.num_classes, 3)
        if not self.use_sigmoid:
            assert seg_label.max().item() == self.num_classes - 1
        else:
            assert seg_label.max().item() == self.num_classes
        valid_vote_preds = vote_preds[vote_mask] # [n_valid, num_cls, 3]
        valid_vote_preds = valid_vote_preds.reshape(-1, 3)
        num_valid = vote_mask.sum()

        valid_label = seg_label[vote_mask]

        if num_valid > 0:
            assert valid_label.max().item() < self.num_classes
            assert valid_label.min().item() >= 0

            indices = torch.arange(num_valid, device=valid_label.device) * self.num_classes + valid_label
            valid_vote_preds = valid_vote_preds[indices, :] #[n_valid, 3]

            valid_vote_targets = vote_targets[vote_mask]

            loss['loss_vote'] = self.loss_vote(valid_vote_preds, valid_vote_targets)
        else:
            loss['loss_vote'] = vote_preds.sum() * 0

        # is_right = seg_logit.argmax(1) == seg_label
        # loss['acc'] = is_right.sum().float() / len(is_right)
        train_cfg = self.train_cfg
        if train_cfg.get('score_thresh', None) is not None:
            score_thresh = train_cfg['score_thresh']
            if self.use_sigmoid:
                scores = seg_logit.sigmoid()
                for i in range(len(score_thresh)):
                    thr = score_thresh[i]
                    name = train_cfg['class_names'][i]
                    this_scores = scores[:, i]
                    pred_true = this_scores > thr
                    real_true = seg_label == i
                    tp = (pred_true & real_true).sum().float()
                    loss[f'recall_{name}'] = tp / (real_true.sum().float() + 1e-5)
            else:
                score = seg_logit.softmax(1)
                group_lens = train_cfg['group_lens']
                group_score = self.gather_group(score[:, :-1], group_lens)
                # pred_true = argmax < 26
                num_fg = score.new_zeros(1)
                for gi in range(len(group_lens)):
                    pred_true = group_score[:, gi] > score_thresh[gi]
                    num_fg += pred_true.sum().float()
                    for i in range(group_lens[gi]):
                        name = train_cfg['group_names'][gi][i]
                        real_true = seg_label == train_cfg['class_names'].index(name)
                        tp = (pred_true & real_true).sum().float()
                        loss[f'recall_{name}'] = tp / (real_true.sum().float() + 1e-5)
                loss[f'num_fg'] = num_fg

        return loss

    def forward_train(self, inputs, img_metas, pts_semantic_mask, vote_targets, vote_mask, return_preds=False):

        seg_logits, vote_preds = self.forward(inputs)
        losses = self.losses(seg_logits, vote_preds, pts_semantic_mask, vote_targets, vote_mask)
        if return_preds:
            return losses, dict(seg_logits=seg_logits, vote_preds=vote_preds)
        else:
            return losses

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

    def get_targets(self, points_list, gt_bboxes_list, gt_labels_list):
        bsz = len(points_list)
        label_list = []
        vote_target_list = []
        vote_mask_list = []
        assert bsz < 10
        cfg = self.train_cfg if self.training else self.test_cfg

        for i in range(bsz):

            points = points_list[i][:, :3]
            bboxes = gt_bboxes_list[i]
            bbox_labels = gt_labels_list[i]

            # if self.num_classes < 3: # I don't know why there are some -1 labels when train car-only model.
            valid_gt_mask = bbox_labels >= 0
            bboxes = bboxes[valid_gt_mask]
            bbox_labels = bbox_labels[valid_gt_mask]
            
            if len(bbox_labels) == 0:
                this_label = torch.ones(len(points), device=points.device, dtype=torch.long) * self.bg_label
                this_vote_target = torch.zeros_like(points)
                vote_mask = torch.zeros_like(this_label).bool()
            else:
                extra_width = self.train_cfg.get('extra_width', None) 
                if extra_width is not None:
                    bboxes = bboxes.enlarged_box_hw(extra_width)
                inbox_inds = bboxes.points_in_boxes(points).long()
                this_label = self.get_point_labels(inbox_inds, bbox_labels)
                this_vote_target, vote_mask = self.get_vote_target(inbox_inds, points, bboxes)

            label_list.append(this_label)
            vote_target_list.append(this_vote_target)
            vote_mask_list.append(vote_mask)

        labels = torch.cat(label_list, dim=0)
        vote_targets = torch.cat(vote_target_list, dim=0)
        vote_mask = torch.cat(vote_mask_list, dim=0)

        return labels, vote_targets, vote_mask
    

    def get_point_labels(self, inbox_inds, bbox_labels):

        bg_mask = inbox_inds < 0
        label = -1 * torch.ones(len(inbox_inds), dtype=torch.long, device=inbox_inds.device)
        class_labels = bbox_labels[inbox_inds]
        class_labels[bg_mask] = self.bg_label
        return class_labels

    def get_vote_target(self, inbox_inds, points, bboxes):

        bg_mask = inbox_inds < 0
        if self.train_cfg.get('centroid_offset', False):
            centroid, _, inv = scatter_v2(points, inbox_inds, mode='avg', return_inv=True)
            center_per_point = centroid[inv]
        else:
            center_per_point = bboxes.gravity_center[inbox_inds]
        delta = center_per_point.to(points.device) - points
        delta[bg_mask] = 0
        target = self.encode_vote_targets(delta)
        vote_mask = ~bg_mask
        return target, vote_mask
    
    def encode_vote_targets(self, delta):
        return torch.sign(delta) * (delta.abs() ** 0.5) 
    
    def decode_vote_targets(self, preds):
        return preds * preds.abs()

@HEADS.register_module()
class FlowSegHead(BaseModule):

    def __init__(self,
                 in_channel,
                 num_classes,
                 hidden_dims=[],
                 dropout_ratio=0.5,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='naiveSyncBN1d'),
                 act_cfg=dict(type='ReLU'),
                 loss_flow=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_obj_vote=None,
                 init_bias=None,
                 init_cfg=None):
        end_channel = hidden_dims[-1] if len(hidden_dims) > 0 else in_channel
        super().__init__()

        self.pre_seg_conv = None
        if len(hidden_dims) > 0:
            self.pre_seg_conv = build_mlp(in_channel, hidden_dims, norm_cfg, act=act_cfg['type'])
        self.num_classes = num_classes
        

        self.use_sigmoid = loss_flow.get('use_sigmoid', False)
        if not self.use_sigmoid:
            self.num_classes += 1

        if loss_flow['type'] == 'FocalLoss':
            self.loss_flow = build_det_loss(loss_flow) # mmdet has a better focal loss supporting single class
        else:
            self.loss_flow = build_loss(loss_flow)

        self.loss_type = loss_flow['type']

        self.pred_linear = nn.Linear(end_channel, self.num_classes)
        if loss_obj_vote is not None:
            self.vote_linear = nn.Linear(end_channel, 3)
            self.loss_obj_vote = build_det_loss(loss_obj_vote)
        self.fp16_enabled = False

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        normal_init(self.pred_linear, mean=0, std=0.01)


    @auto_fp16(apply_to=('pts_feat',))
    def forward(self, pts_feat):
        """Forward pass.

        """
        out_dict = {}

        output = pts_feat
        if self.pre_seg_conv is not None:
            output = self.pre_seg_conv(pts_feat)
        out_dict['moving_pred'] = self.pred_linear(output)
        if self.train_cfg.get('voting_objectness', False):
            out_dict['voting_pred'] = self.vote_linear(output)
        return out_dict

    @force_fp32(apply_to=('pred', 'voting_label'))
    def losses(self, pred, is_moving_label, voting_label, voting_mask):
        loss = dict()
        loss['loss_flow'] = self.loss_flow(pred['moving_pred'], is_moving_label)

        if self.train_cfg.get('voting_objectness', False):
            voting_pred = pred['voting_pred']
            if voting_mask.any():
                loss['loss_obj_vote'] = self.loss_obj_vote(voting_pred[voting_mask], voting_label)
            else:
                loss['loss_obj_vote'] = voting_pred.sum() * 0

        return loss

    def forward_train(self, inputs, img_metas, is_moving_label, voting_label, voting_mask, gt_bboxes_3d=None, gt_labels_3d=None, voxel_info=None):

        forward_out = self.forward(inputs)
        losses = self.losses(forward_out, is_moving_label, voting_label, voting_mask)

        if self.train_cfg.get('affinity'):
            affinity_loss = self.get_affinity_loss(voxel_info, gt_bboxes_3d, gt_labels_3d)
            losses.update(affinity_loss)

        losses['num_moving_pts'] = (is_moving_label == 0).sum().float()

        if self.loss_type in ['CrossEntropyLoss', 'FocalLoss']:
            scores = forward_out['moving_pred'].clone().detach().sigmoid()
            moving_score_thr = self.train_cfg.get('moving_score_thr', 0.5)
            pred_moving = (scores > moving_score_thr).reshape(-1)
            assert is_moving_label.ndim == 1
            tp = (pred_moving & (is_moving_label == 0)).sum().float()
            losses['Acc'] = tp / (pred_moving.sum().float() + 1e-5)
            losses['Recall'] = tp / ((is_moving_label == 0).sum().float() + 1e-5)


        return losses
    
    def get_targets(self, raw_flow, points, gt_bboxes_3d):
        bsz = len(raw_flow)
        assert isinstance(gt_bboxes_3d, list)
        assert isinstance(raw_flow, list)
        assert bsz < 10
        cfg = self.train_cfg
        if self.loss_type in ['CrossEntropyLoss', 'FocalLoss']:
            thr = self.train_cfg['moving_thr']
        else:
            raise NotImplementedError
        
        moving_target_list = []
        voting_valid_mask_list = []
        is_moving_list = []
        voting_target_list = []
        for i in range(bsz):
            is_moving = (raw_flow[i].abs() > thr).any(1)
            moving_targets = torch.ones_like(is_moving, dtype=torch.long) # 1 is not moving (background)
            moving_targets[is_moving] = 0
            is_moving_list.append(is_moving)
            moving_target_list.append(moving_targets)
        
        if cfg.get('voting_objectness', False):
            for i in range(bsz):
                boxes = gt_bboxes_3d[i].noisy_box(cfg['center_noise'], cfg['dim_noise'], cfg['yaw_noise'])
                inbox_inds = boxes.points_in_boxes(points[i]).long()
                valid_mask = is_moving_list[i] & (inbox_inds > -1)
                voting_valid_mask_list.append(valid_mask)

                if valid_mask.any():
                    valid_pts = points[i][valid_mask]
                    valid_inds = inbox_inds[valid_mask]
                    center_per_pts = boxes.gravity_center[valid_inds]
                    voting_target = center_per_pts - valid_pts
                    voting_target_list.append(voting_target)
        
        is_moving_label = torch.cat(moving_target_list, dim=0)

        if len(voting_target_list) > 0:
            voting_label = torch.cat(voting_target_list, dim=0)
        else:
            voting_label = None
        
        if len(voting_valid_mask_list) > 0:
            voting_mask = torch.cat(voting_valid_mask_list, dim=0)
        else:
            voting_mask = None

        return is_moving_label, voting_label, voting_mask
    
    def get_affinity_loss(self, voxel_info, gt_bboxes_3d, gt_labels_3d):
        voxel_feats = voxel_info['voxel_feats']
        voxel_coors = voxel_info['voxel_coors']
        voxel_feats = self.affinity_proj(voxel_feats)
        voxel_feats = torch.nn.functional.normalize(voxel_feats, dim=1)
        voxel_feats, voxel_coors, voxel_centers, voxel_gt_inds = self.sample_voxel_for_affinity(voxel_feats, voxel_coors)
        affinity_mat = (voxel_feats @ voxel_feats.T + 1) / 2
        is_in_same_ins = voxel_gt_inds[:, None] == voxel_gt_inds[None, :]
        dis_mat = torch.linalg.norm(voxel_centers[:, None] - voxel_centers[None, :], dim=1, ord=2)
        valid_for_loss = dis_mat < self.train_cfg['valid_affinity_distance', 10]
        
    
    def sample_voxel_for_affinity(self, voxel_feats, voxel_coors, gt_bboxes_3d):
        device = voxel_feats.device
        bsz = len(gt_bboxes_3d)
        assert isinstance(gt_bboxes_3d, list)
        gt_bboxes_3d = gt_bboxes_3d[0].enlarged_box(self.train_cfg.get('enlarge_width_for_affinity', 1))
        assert bsz == voxel_coors[:, 0].max().item() + 1
        assert bsz == 1

        max_voxel_num = self.train_cfg.get('max_affinity_dim', 500)

        voxel_size = torch.tensor(self.voxel_size, dtype=dtype, device=device).reshape(1,3)
        pc_min_range = torch.tensor(self.point_cloud_range[:3], dtype=dtype, device=device).reshape(1,3)
        voxel_centers = (voxel_coors[:, [3,2,1]].to(dtype).to(device) + 0.5) * voxel_size + pc_min_range# x y z order

        gt_inds = gt_bboxes_3d.points_in_boxes(voxel_centers)

        valid_mask = gt_inds > -1
        if not valid_mask.any():
            return None, None, None, None

        valid_feats = voxel_feats[valid_mask]
        valid_coors = voxel_coors[valid_mask]
        valid_centers = voxel_centers[valid_mask]
        valid_inds = gt_inds[valid_mask]

        voxel_num = len(valid_feats)
        if voxel_num <= max_voxel_num:
            return valid_feats, valid_coors, valid_centers, valid_inds
        
        shfl_inds = torch.randperm(voxel_num, device=device)[:max_voxel_num]

        return valid_feats[shfl_inds], valid_coors[shfl_inds], valid_centers[shfl_inds], valid_inds[shfl_inds]



        
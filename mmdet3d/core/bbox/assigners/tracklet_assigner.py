import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner

@BBOX_ASSIGNERS.register_module()
class TrackletAssigner(BaseAssigner):

    def __init__(self, object_centric=False, iou_thr=0.5):
        self.object_centric = object_centric
        self.iou_thr = iou_thr
        return

    def assign(self, trk_pd, trk_gt):

        device = trk_pd.device
        num_gts, num_bboxes = len(trk_gt), len(trk_pd)
        gt_labels = torch.full((num_gts,), trk_gt.type, dtype=torch.long, device=device) 

        # 1. assign -1 by default
        assigned_labels = gt_labels.new_full((num_bboxes, ), -1)

        if num_gts == 0 or num_bboxes == 0:
            assigned_gt_inds = torch.full((num_bboxes, ), -1, dtype=torch.long, device=device)
            # No ground truth or boxes, return empty assignment
            max_overlaps = torch.full((num_bboxes, ), 0, dtype=torch.float, device=device)
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0

            assign_result = AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
            scores = trk_pd.concated_scores().detach()
            assign_result.scores = scores
            assert len(max_overlaps) == len(scores)
            return assign_result

        overlaps = trk_pd.self_ious(trk_gt)
        scores = trk_pd.concated_scores().detach()

        if self.object_centric:
            assigned_gt_inds = torch.tensor(
                [trk_gt.get_index_from_ts(ts) + 1  if overlaps[i].item() > self.iou_thr else 0 for i, ts in enumerate(trk_pd.ts_list)],
                dtype=torch.long, device=device
            )
        else:
            assigned_gt_inds = torch.tensor([trk_gt.get_index_from_ts(ts) + 1 for ts in trk_pd.ts_list], dtype=torch.long, device=device)

        assert (assigned_gt_inds >= 0).all()
        assigned_labels[assigned_gt_inds > 0] = trk_gt.type

        assert len(overlaps) == len(scores)

        assign_result = AssignResult(num_gts, assigned_gt_inds, overlaps, labels=assigned_labels)
        assign_result.scores = scores
        return assign_result
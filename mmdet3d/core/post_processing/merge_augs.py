import torch

from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu, weighted_nms
from ..bbox import bbox3d2result, bbox3d_mapping_back, xywhr2xyxyr
from ipdb import set_trace


def merge_aug_bboxes_3d(aug_results, img_metas, test_cfg):
    """Merge augmented detection 3D bboxes and scores.

    Args:
        aug_results (list[dict]): The dict of detection results.
            The dict contains the following keys

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
        img_metas (list[dict]): Meta information of each sample.
        test_cfg (dict): Test config.

    Returns:
        dict: Bounding boxes results in cpu mode, containing merged results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Merged detection bbox.
            - scores_3d (torch.Tensor): Merged detection scores.
            - labels_3d (torch.Tensor): Merged predicted box labels.
    """

    assert len(aug_results) == len(img_metas), \
        '"aug_results" should have the same length as "img_metas", got len(' \
        f'aug_results)={len(aug_results)} and len(img_metas)={len(img_metas)}'

    recovered_bboxes = []
    recovered_scores = []
    recovered_labels = []

    for bboxes, img_info in zip(aug_results, img_metas):
        scale_factor = img_info[0]['pcd_scale_factor']
        pcd_horizontal_flip = img_info[0]['pcd_horizontal_flip']
        pcd_vertical_flip = img_info[0]['pcd_vertical_flip']
        recovered_scores.append(bboxes['scores_3d'])
        recovered_labels.append(bboxes['labels_3d'])
        bboxes = bbox3d_mapping_back(bboxes['boxes_3d'], scale_factor,
                                     pcd_horizontal_flip, pcd_vertical_flip)
        recovered_bboxes.append(bboxes)

    aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
    aug_bboxes_for_nms = xywhr2xyxyr(aug_bboxes.bev)
    aug_scores = torch.cat(recovered_scores, dim=0)
    aug_labels = torch.cat(recovered_labels, dim=0)

    # TODO: use a more elegent way to deal with nms
    if test_cfg.use_rotate_nms:
        nms_func = nms_gpu
    else:
        nms_func = nms_normal_gpu

    merged_bboxes = []
    merged_scores = []
    merged_labels = []

    # Apply multi-class nms when merge bboxes
    if len(aug_labels) == 0:
        return bbox3d2result(aug_bboxes, aug_scores, aug_labels)

    for class_id in range(torch.max(aug_labels).item() + 1):
        class_inds = (aug_labels == class_id)
        bboxes_i = aug_bboxes[class_inds]
        bboxes_nms_i = aug_bboxes_for_nms[class_inds, :]
        scores_i = aug_scores[class_inds]
        labels_i = aug_labels[class_inds]
        if len(bboxes_nms_i) == 0:
            continue
        selected = nms_func(bboxes_nms_i, scores_i, test_cfg.nms_thr)

        merged_bboxes.append(bboxes_i[selected, :])
        merged_scores.append(scores_i[selected])
        merged_labels.append(labels_i[selected])

    merged_bboxes = merged_bboxes[0].cat(merged_bboxes)
    merged_scores = torch.cat(merged_scores, dim=0)
    merged_labels = torch.cat(merged_labels, dim=0)

    _, order = merged_scores.sort(0, descending=True)
    num = min(test_cfg.max_num, len(aug_bboxes))
    order = order[:num]

    merged_bboxes = merged_bboxes[order]
    merged_scores = merged_scores[order]
    merged_labels = merged_labels[order]

    return bbox3d2result(merged_bboxes, merged_scores, merged_labels)

def inverse_aug(boxes, img_meta):
    pcd_horizontal_flip = img_meta['pcd_horizontal_flip']
    pcd_vertical_flip = img_meta['pcd_vertical_flip']
    rot_angle = img_meta.get('pcd_rot_angle', 0)

    if pcd_horizontal_flip:
        boxes.flip('horizontal')

    if pcd_vertical_flip:
        boxes.flip('vertical')

    if rot_angle != 0:
        boxes.rotate(rot_angle) # rotation of points is opposite to the rotation of bboxes
    return boxes

def weighted_box_fusion(boxes_bev, scores, labels, lidar_boxes, nms_thr, merge_thr, tta_cfg):
    data2merge = lidar_boxes.tensor.clone()
    if tta_cfg.get('mean_score', False):
        data2merge = torch.cat([data2merge, scores[:, None]], 1)
        
    keep, out_data, cnt = weighted_nms(boxes_bev, data2merge, scores, nms_thr, merge_thr)
    assert out_data.shape[-1] == data2merge.shape[-1] + 1, 'One more dimension for max scores'
    out_xyzwlh = out_data[:, :6]
    out_yaws = lidar_boxes.tensor[keep, -1:]
    out_boxes = torch.cat([out_xyzwlh, out_yaws], 1)
    assert out_boxes.shape[-1] == 7

    if tta_cfg.get('mean_score', False):
        assert out_data.shape[-1] == 9
        out_scores = out_data[:, -2]
    else:
        assert out_data.shape[-1] == 8
        out_scores = out_data[:, -1]

    out_labels = labels[keep]
    out_boxes = type(lidar_boxes)(out_boxes)
    return out_boxes, out_scores, out_labels

def merge_augs_better(aug_results, img_metas, tta_cfg, device):

    assert len(aug_results) == len(img_metas), \
        '"aug_results" should have the same length as "img_metas", got len(' \
        f'aug_results)={len(aug_results)} and len(img_metas)={len(img_metas)}'

    recovered_bboxes = []
    recovered_scores = []
    recovered_labels = []

    for result, img_info in zip(aug_results, img_metas):
        boxes, scores, labels = result['boxes_3d'], result['scores_3d'], result['labels_3d']
        boxes.tensor = boxes.tensor.to(device)

        boxes = inverse_aug(boxes, img_info[0])

        recovered_bboxes.append(boxes)
        recovered_scores.append(scores)
        recovered_labels.append(labels)

    aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
    aug_bboxes_for_nms = xywhr2xyxyr(aug_bboxes.bev)
    aug_scores = torch.cat(recovered_scores, dim=0)
    aug_labels = torch.cat(recovered_labels, dim=0)

    # TODO: use a more elegent way to deal with nms
    nms_func = nms_gpu

    merged_bboxes = []
    merged_scores = []
    merged_labels = []

    # Apply multi-class nms when merge bboxes
    if len(aug_labels) == 0:
        return bbox3d2result(aug_bboxes, aug_scores, aug_labels)
    
    nms_thr = tta_cfg.nms_thr
    if isinstance(nms_thr, (float, int)):
        nms_thr = [nms_thr, ] * tta_cfg.num_classes

    for class_id in range(tta_cfg.num_classes):
        class_inds = (aug_labels == class_id)
        bboxes_i = aug_bboxes[class_inds]
        bboxes_nms_i = aug_bboxes_for_nms[class_inds, :]
        scores_i = aug_scores[class_inds]
        labels_i = aug_labels[class_inds]
        if len(bboxes_nms_i) == 0:
            continue
        if tta_cfg.get('wnms', False):
            scores_i = scores_i.to(device)
            labels_i = labels_i.to(device)
            out_boxes, out_scores, out_labels = weighted_box_fusion(bboxes_nms_i, scores_i, labels_i, bboxes_i, nms_thr[class_id], tta_cfg.wnms_merge_thr[class_id], tta_cfg)
            merged_bboxes.append(out_boxes)
            merged_scores.append(out_scores)
            merged_labels.append(out_labels)
        else:
            selected = nms_func(bboxes_nms_i, scores_i, nms_thr[class_id])
            merged_bboxes.append(bboxes_i[selected, :])
            merged_scores.append(scores_i[selected])
            merged_labels.append(labels_i[selected])

    merged_bboxes = merged_bboxes[0].cat(merged_bboxes)
    merged_scores = torch.cat(merged_scores, dim=0)
    merged_labels = torch.cat(merged_labels, dim=0)

    # _, order = merged_scores.sort(0, descending=True)
    # num = min(test_cfg.max_num, len(aug_bboxes))
    # order = order[:num]

    # merged_bboxes = merged_bboxes[order]
    # merged_scores = merged_scores[order]
    # merged_labels = merged_labels[order]

    return bbox3d2result(merged_bboxes, merged_scores, merged_labels)


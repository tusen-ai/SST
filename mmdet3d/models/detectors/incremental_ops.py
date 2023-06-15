import torch
import torch.nn.functional as F
import os
from ipdb import set_trace
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.utils import TorchTimer
timer = TorchTimer(-1)
try:
    from torchex import incremental_points_mask
except ImportError:
    incremental_points_mask = None

def voxelize_single(points, voxel_size, pc_range):
    res = points
    res_coors = torch.div(res[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').int() # I guess int may be faster than long type.
    # res_coors = res_coors[:, [2, 1, 0]] # to zyx order
    return res_coors

def find_delta_unq_coors(unq1, unq2):
    all_coors = torch.cat([unq1, unq2], dim=0)
    _, inv, counts = torch.unique(all_coors, return_counts=True, return_inverse=True, dim=0)
    duplicate_mask = (counts[inv] > 1)[-len(unq2):]
    delta_coors = unq2[~duplicate_mask]
    # assert len(torch.unique(unq2, dim=0)) == len(torch.unique(torch.cat([unq2, delta_coors], dim=0), dim=0))
    return delta_coors

def find_delta_pc_mask(all_coors, sub_coors):
    _, old_inv, old_counts = torch.unique(all_coors, sorted=True, return_counts=True, return_inverse=True, dim=0)

    _, inv, counts = torch.unique(torch.cat([all_coors, sub_coors], 0), sorted=True, return_counts=True, return_inverse=True, dim=0)
    mask = counts > old_counts
    mask = mask[old_inv]
    return mask

# def find_delta_pc_mask_v2(all_coors, old_counts, old_inv, sub_coors):
#     _, old_inv, old_counts = torch.unique(all_coors, sorted=True, return_counts=True, return_inverse=True, dim=0)

#     _, inv, counts = torch.unique(torch.cat([all_coors, sub_coors], 0), sorted=True, return_counts=True, return_inverse=True, dim=0)
#     mask = counts > old_counts
#     mask = mask[old_inv]
#     return mask


def find_delta_points_by_voxelization(pc1, pc2, voxel_size, pc_range):
    """
    return the delta points in pc2
    """
    device = pc1.device
    voxel_size = torch.tensor(voxel_size, device=device)
    pc_range = torch.tensor(pc_range, device=device)
    coors1 = voxelize_single(pc1, voxel_size, pc_range)
    coors2 = voxelize_single(pc2, voxel_size, pc_range)
    # unq_coors1 = torch.unique(coors1, dim=0)
    # unq_coors2 = torch.unique(coors2, dim=0)
    # delta_unq_coors = find_delta_unq_coors(unq_coors1, unq_coors2)
    # delta_pc_mask = find_delta_pc_mask(coors2, delta_unq_coors) 
    xs = int((pc_range[3] - pc_range[0]) / voxel_size[0]) + 1
    ys = int((pc_range[4] - pc_range[1]) / voxel_size[1]) + 1
    zs = int((pc_range[5] - pc_range[2]) / voxel_size[2]) + 1
    spatial_size = (xs, ys, zs)
    mask = incremental_points_mask(coors1, coors2, spatial_size)
    return pc2[mask]

def find_delta_points_by_voxelization_list(pc1_list, pc2, voxel_size, pc_range):
    """
    return the delta points in pc2
    """
    device = pc1_list[0].device
    voxel_size = torch.tensor(voxel_size, device=device)
    pc_range = torch.tensor(pc_range, device=device)
    coors1_list = [voxelize_single(pc1, voxel_size, pc_range) for pc1 in pc1_list]
    unq_coors1_list = [torch.unique(c, dim=0) for c in coors1_list]
    coors1 = torch.unique(torch.cat(unq_coors1_list, 0), dim=0) 
    coors2 = voxelize_single(pc2, voxel_size, pc_range)
    # unq_coors1 = torch.unique(coors1, dim=0)
    unq_coors1 = coors1
    unq_coors2 = torch.unique(coors2, dim=0)
    delta_unq_coors = find_delta_unq_coors(unq_coors1, unq_coors2)
    delta_pc_mask = find_delta_pc_mask(coors2, delta_unq_coors) 
    return pc2[delta_pc_mask]

def find_delta_points_by_voxelization_list_v2(pc1_list, pc2, voxel_size, pc_range):
    """
    a little faster than v1
    """
    device = pc1_list[0].device
    voxel_size = torch.tensor(voxel_size, device=device)
    pc_range = torch.tensor(pc_range, device=device)
    unq_coors1 = torch.unique(voxelize_single(torch.cat(pc1_list, 0), voxel_size, pc_range), dim=0) 
    coors2 = voxelize_single(pc2, voxel_size, pc_range)
    unq_coors2, inv_coors2 = torch.unique(coors2, return_inverse=True, dim=0)
    delta_unq_coors = find_delta_unq_coors(unq_coors1, unq_coors2)
    _, _, counts = torch.unique(torch.cat([unq_coors2, delta_unq_coors], 0), sorted=True, return_counts=True, return_inverse=True, dim=0)
    mask = counts > 1
    mask = mask[inv_coors2]
    return pc2[mask]

def find_delta_points_by_voxelization_list_v3(pc1_list, pc2, voxel_size, pc_range):
    """
    a much faster version
    """

    device = pc1_list[0].device
    xs = int((pc_range[3] - pc_range[0]) / voxel_size[0]) + 1
    ys = int((pc_range[4] - pc_range[1]) / voxel_size[1]) + 1
    zs = int((pc_range[5] - pc_range[2]) / voxel_size[2]) + 1
    spatial_size = (xs, ys, zs)
    voxel_size = torch.tensor(voxel_size, device=device)
    pc_range = torch.tensor(pc_range, device=device)
    pc1 = torch.cat(pc1_list, 0)

    # should be larger than lower bound to make sure voxel_coors >= 0
    in_range_mask1 = (pc1[:, 0] > pc_range[0]) & (pc1[:, 1] > pc_range[1]) & (pc1[:, 2] > pc_range[2]) 
    in_range_mask2 = (pc2[:, 0] > pc_range[0]) & (pc2[:, 1] > pc_range[1]) & (pc2[:, 2] > pc_range[2]) 
    pc1 = pc1[in_range_mask1]
    pc2 = pc2[in_range_mask2]

    coors1 = voxelize_single(pc1, voxel_size, pc_range)
    coors2 = voxelize_single(pc2, voxel_size, pc_range)
    mask = incremental_points_mask(coors1, coors2, spatial_size)
    out = pc2[mask]
    return out

def box_frame_transform_gpu(pre_boxes, pre_pose, cur_pose, cur_pose_inv=None):
    assert isinstance(pre_boxes, LiDARInstance3DBoxes)
    pre_boxes_lidar = pre_boxes
    pre_boxes = pre_boxes_lidar.tensor
    assert pre_boxes.size(1) in (7, 9)

    if len(pre_boxes) == 0:
        return pre_boxes_lidar

    pre_centers = pre_boxes[:, :3]
    pre_centers_h = F.pad(pre_centers, (0, 1), 'constant', 1)
    heading_vector = pre_boxes_lidar.heading_unit_vector
    heading_vector_h = F.pad(heading_vector, (0, 1), 'constant', 1) 

    if cur_pose_inv is None:
        world2curr_pose = torch.linalg.inv(cur_pose)
    else:
        world2curr_pose = cur_pose_inv

    mm = world2curr_pose @ pre_pose
    # centers_in_curr = torch.einsum('ij,nj->ni', mm, pre_centers_h)[:, :3]
    centers_in_curr = (pre_centers_h @ mm.T)[:, :3]

    mm_zero_t = mm.clone()
    mm_zero_t[:3, 3] = 0 # a math trick
    # heading_vector_in_curr = torch.einsum('ij,nj->ni', mm_zero_t, heading_vector_h)[:, :3]
    heading_vector_in_curr = (heading_vector_h @ mm_zero_t.T)[:, :3]
    yaw_in_curr = torch.atan2(heading_vector_in_curr[:, 0], heading_vector_in_curr[:, 1])

    transformed_boxes = torch.cat([centers_in_curr, pre_boxes[:, 3:6], yaw_in_curr[:, None]], axis=1)
    if pre_boxes.size(1) == 9:
        velo = pre_boxes[:, [7, 8]]
        velo = F.pad(velo, (0, 1), 'constant', 0) # pad zeros as z-axis velocity
        velo = velo @ mm[:3, :3].T
        transformed_boxes = torch.cat([transformed_boxes, velo[:, :2]], dim=1)

    transformed_boxes = LiDARInstance3DBoxes(transformed_boxes, box_dim=transformed_boxes.size(1))
    return transformed_boxes

# def points_frame_transform(past_points, pre_pose, cur_pose):
#     past2world_rot = pre_pose[0:3, 0:3]
#     past2world_trans = pre_pose[0:3, 3]
#     world2curr_pose = torch.inverse(cur_pose)
#     world2curr_rot = world2curr_pose[0:3, 0:3]
#     world2curr_trans = world2curr_pose[0:3, 3]
#     set_trace()
#     past_pc_in_world = torch.matmul(past_points, past2world_rot.T) + past2world_trans[None, :]
#     past_pc_in_curr = torch.matmul(past_pc_in_world,world2curr_rot.T) + world2curr_trans[None, :]
#     return past_pc_in_curr

def points_frame_transform(pre_points, pre_pose, cur_pose, cur_pose_inv=None):
    pre_points_h = torch.nn.functional.pad(pre_points, (0, 1), 'constant', 1)

    if cur_pose_inv is None:
        world2curr_pose = torch.inverse(cur_pose)
    else:
        world2curr_pose = cur_pose_inv

    mm = world2curr_pose @ pre_pose
    # pre_points_in_cur = torch.einsum('ij,nj->ni', mm, pre_points_h)[:, :3]
    pre_points_in_cur = (pre_points_h @ mm.T)[:, :3]
    return pre_points_in_cur

def generate_virtual_seed_points(seed, cfg=None):
    boxes = seed['gt_bboxes_3d']
    labels = seed['gt_labels_3d']
    scores = seed['scores']
    assert len(boxes.tensor) == len(labels)
    if len(boxes.tensor) == 0:
        return None, None, None, None
    cano_bev_corners = boxes.canonical_corners[:, [1, 3, 7, 4], :2] # [N, 4, 2]
    device = cano_bev_corners.device
    unq_labels = torch.unique(labels).tolist()

    vertex_list = []
    center_offset_list = []
    label_list = []
    score_list = []

    for cls in unq_labels:
        cls_mask = labels == cls
        if not cls_mask.any():
            continue
        this_cano_bev_corners = cano_bev_corners[cls_mask]
        this_boxes_tensor = boxes.tensor[cls_mask]

        l_corners = this_cano_bev_corners[:, [0, 1]] #[num_boxes, 2, 2]
        num_interval = cfg[cls]['length_vertices'] + 1
        delta = (l_corners[:, 1, 1] - l_corners[:, 0, 1]) / num_interval # [num_boxes,]
        new_delta_y = torch.arange(num_interval + 1, device=device).float()[None, :] * delta[:, None] # [num_boxes, num_interval+1]
        new_delta_x = torch.zeros_like(new_delta_y)
        new_delta = torch.stack([new_delta_x, new_delta_y], -1) #[num_boxes, num_interval+1, 2]
        vertex_in_length = l_corners[:, [0,], :] + new_delta #[num_boxes, num_interval+1, 2]

        w_corners = this_cano_bev_corners[:, [1, 2]] #[num_boxes, 2, 2]
        num_interval = cfg[cls]['width_vertices'] + 1
        delta = (w_corners[:, 1, 0] - w_corners[:, 0, 0]) / num_interval # [num_boxes,]
        new_delta_x = torch.arange(num_interval + 1, device=device).float()[None, :] * delta[:, None] # [num_boxes, num_interval+1]
        new_delta_y = torch.zeros_like(new_delta_x)
        new_delta = torch.stack([new_delta_x, new_delta_y], -1) #[num_boxes, num_interval+1, 2]
        vertex_in_width = w_corners[:, [0,], :] + new_delta #[num_boxes, num_interval+1, 2]

        # copy the vertex in two edges
        widths = this_boxes_tensor[:, [3]]
        vertex_in_length_copy1 = vertex_in_length.clone()
        vertex_in_length_copy2 = vertex_in_length.clone()
        vertex_in_length_copy1[:, :, 0] += widths / 2
        vertex_in_length_copy2[:, :, 0] += widths

        lengths = this_boxes_tensor[:, [4]]
        vertex_in_width_copy1 = vertex_in_width.clone()
        vertex_in_width_copy2 = vertex_in_width.clone()
        vertex_in_width_copy1[:, :, 1] -= lengths / 2
        vertex_in_width_copy2[:, :, 1] -= lengths

        bev_vertex = torch.cat([
            vertex_in_length,
            vertex_in_length_copy1,
            vertex_in_length_copy2,
            vertex_in_width,
            vertex_in_width_copy1,
            vertex_in_width_copy2,
            torch.zeros_like(vertex_in_length),
        ], 1) #[num_boxes, num_vertex, 2]
        num_bev_vertices = bev_vertex.size(1)

        top_z = this_boxes_tensor[:, 5] # box heights
        top_z = top_z[:, None, None].expand(-1, num_bev_vertices, -1)
        bot_z = torch.zeros_like(top_z)

        top_vertex = torch.cat([bev_vertex, top_z], -1)
        bot_vertex = torch.cat([bev_vertex, bot_z], -1)
        all_vertex = torch.cat([top_vertex, bot_vertex], 1)

        center_offset = all_vertex.clone() # canonical center offset
        center_offset[:, :, [2]] -= torch.cat([top_z, top_z], 1) / 2

        all_vertex = rotation_3d_in_axis(all_vertex, this_boxes_tensor[:, 6], axis=2)
        all_vertex += this_boxes_tensor[:, None, :3]

        if scores is not None:
            num_vertex_per_box = all_vertex.size(1)
            this_scores = scores[cls_mask, None].expand(-1, num_vertex_per_box).reshape(-1)
            score_list.append(this_scores)

        all_vertex = all_vertex.reshape(-1, 3)
        center_offset = center_offset.reshape(-1, 3)

        vertex_list.append(all_vertex)
        center_offset_list.append(center_offset)

        label_list.append(all_vertex.new_ones(len(all_vertex)) * cls)

        out_virtual_points = torch.cat(vertex_list, 0)
        out_center_offsets = torch.cat(center_offset_list, 0)
        out_labels = torch.cat(label_list, 0)
        out_scores = torch.cat(score_list, 0) if scores is not None else None

        x, y, z = out_virtual_points[:, 0], out_virtual_points[:, 1], out_virtual_points[:, 2]
        pc_range = cfg['point_cloud_range']
        eps = 1e-3
        mask = (
              (x > pc_range[0] + eps)
            & (y > pc_range[1] + eps)
            & (z > pc_range[2] + eps)
            & (x < pc_range[3] - eps)
            & (y < pc_range[4] - eps)
            & (z < pc_range[5] - eps)
        )


        return out_virtual_points[mask], out_center_offsets[mask], out_labels[mask], out_scores[mask]
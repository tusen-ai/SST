import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
from mmdet3d.core import LiDARInstance3DBoxes
from ipdb import set_trace

COLOR_MAP = {
    'gray': np.array([140, 140, 136]) / 256,
    'light_blue': np.array([4, 157, 217]) / 256,
    'blue': np.array([0, 0, 255]) / 256,
    'wine_red': np.array([191, 4, 54]) / 256,
    'red': np.array([255, 0, 0]) / 256,
    'black': np.array([0, 0, 0]) / 256,
    'purple': np.array([224, 133, 250]) / 256, 
    'dark_green': np.array([32, 64, 40]) / 256,
    'green': np.array([77, 115, 67]) / 256,
    'yellow': np.array([255, 255, 0]) / 256
}

def vis_voxel_label(name, voxel_coors, label, voxel_size=[0.32, 0.32, 5], pc_range=[-51.2, -51.2, -3, 51.2, 51.2, 2], resolution=(2048, 2028), dir=None, root=None):
    '''
    voxel_size=(0.2, 0.2, 0.5),
    point_cloud_range=(-20, -20, -3, 20, 20, 3),
    coords: [N, 3] (b, z, y, x)
    resolution: (num_pxl_x, num_pxl_y)
    '''
    if isinstance(voxel_coors, torch.Tensor):
        voxel_coors = voxel_coors.detach().cpu().numpy().astype(np.float)
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    assert voxel_coors.shape[0] == label.shape[0]

    assert (voxel_coors[:, 0] == 0).all()

    pxl_per_meter = resolution[0] / (pc_range[3] - pc_range[0])

    voxel_size = np.array(voxel_size) 
    voxel_coors = voxel_coors[:, [3,2,1]] # [x, y, z]
    points = (voxel_coors + 0.5) * voxel_size[None, :] + np.array(pc_range[:3])
    # to pixel coordinates

    x = ((points[:, 0] - pc_range[0])  * pxl_per_meter).astype(np.int32)
    y = ((points[:, 1] - pc_range[1]) * pxl_per_meter).astype(np.int32)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlim(0, resolution[1])
    ax.set_xlabel('Y')
    ax.set_ylim(0, resolution[0])
    ax.set_xlabel('X')
    ax.set_aspect('equal')

    unique_labels = np.unique(label)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(unique_labels))))

    data_cfg = yaml.safe_load(open('/mnt/truenas/scratch/lve.fan/semantic-kitti-api/config/semantic-kitti.yaml', 'r')) #[b, g, r]
    color_map = data_cfg['color_map']
    learn_map_inv = data_cfg['learning_map_inv']

    for unq_label in unique_labels:
        this_mask = label == unq_label
        this_x = x[this_mask]
        this_y = y[this_mask]
        num_points = len(this_x)
        color = np.array(color_map[learn_map_inv[int(unq_label)]])[[2, 1, 0]] / 255
        color = np.concatenate([color[None,:],] * num_points, 0)
        ax.scatter(this_y, this_x, s=0.1, c=color) # swap x-axis and y-axis due to x is the forward direction in point cloud.

    if root is None:
        root = '/mnt/truenas/scratch/lve.fan/transdet3d/work_dirs_seg/figs'
    if dir:
        root = os.path.join(root, dir)
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    plt.savefig(os.path.join(root, name))
    print(f'Save to {os.path.join(root, name)}')

def vis_point_label(name, points, label, pc_range=[-51.2, -51.2, -3, 51.2, 51.2, 2], resolution=(2048, 2028), dir=None, root=None):
    '''
    voxel_size=(0.2, 0.2, 0.5),
    point_cloud_range=(-20, -20, -3, 20, 20, 3),
    coords: [N, 3] (b, z, y, x)
    resolution: (num_pxl_x, num_pxl_y)
    '''
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy().astype(np.float)
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    assert points.shape[0] == label.shape[0]


    pxl_per_meter = resolution[0] / (pc_range[3] - pc_range[0])

    x = ((points[:, 0] - pc_range[0])  * pxl_per_meter).astype(np.int32)
    y = ((points[:, 1] - pc_range[1]) * pxl_per_meter).astype(np.int32)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlim(0, resolution[1])
    ax.set_xlabel('Y')
    ax.set_ylim(0, resolution[0])
    ax.set_xlabel('X')
    ax.set_aspect('equal')

    unique_labels = np.unique(label)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(unique_labels))))

    data_cfg = yaml.safe_load(open('/mnt/truenas/scratch/lve.fan/semantic-kitti-api/config/semantic-kitti.yaml', 'r')) #[b, g, r]
    color_map = data_cfg['color_map']
    learn_map_inv = data_cfg['learning_map_inv']

    for unq_label in unique_labels:
        this_mask = label == unq_label
        this_x = x[this_mask]
        this_y = y[this_mask]
        num_points = len(this_x)
        color = np.array(color_map[learn_map_inv[int(unq_label)]])[[2, 1, 0]] / 255
        color = np.concatenate([color[None,:],] * num_points, 0)
        ax.scatter(this_y, this_x, s=0.1, c=color) # swap x-axis and y-axis due to x is the forward direction in point cloud.

    if root is None:
        root = '/mnt/truenas/scratch/lve.fan/transdet3d/work_dirs_seg/figs'
    if dir:
        root = os.path.join(root, dir)
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    plt.savefig(os.path.join(root, name))
    print(f'Save to {os.path.join(root, name)}')

from .visualizer import Visualizer2D
def vis_bev_pc(pc, gts=None, pds=None, name='', save_root='./work_dirs/figs', figsize=(40, 40), color='gray', dir=None, messages=None, s=0.1, pc_range=None):
    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().detach().numpy()
    assert '.png' in name
    if dir:
        save_root = os.path.join(save_root, dir)
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, name)

    if pc_range is not None:
        x_range = y_range = (-pc_range, pc_range)
    else:
        x_range = y_range = None

    visualizer = Visualizer2D(name='', figsize=figsize, x_range=x_range, y_range=y_range)
    if pc is not None:
        visualizer.handler_pc(pc, s=s, color=color)

    if isinstance(gts, list):
        # for tracklet visualization
        colors = ['red', 'blue', 'green', 'black', 'light_blue']
        for frame_idx, gt in enumerate(gts):
            if gt is None:
                continue
            assert isinstance(gt, LiDARInstance3DBoxes)
            gt = gt.corners.to('cpu')
            for i in range(len(gt)):
                visualizer.handler_box_4corners(gt[i, [1, 3, 7, 4], :2], color=colors[frame_idx % 5])

    elif gts is not None and len(gts) > 0:
        assert isinstance(gts, LiDARInstance3DBoxes)
        gts = gts.corners.to('cpu')
        for i in range(len(gts)):
            visualizer.handler_box_4corners(gts[i, [1, 3, 7, 4], :2])

    if pds is not None and len(pds) > 0: 
        assert isinstance(pds, LiDARInstance3DBoxes)
        pds = pds.corners.to('cpu')
        for i in range(len(pds)):
            visualizer.handler_box_4corners(pds[i, [1, 3, 7, 4], :2], message='' if messages is None else messages[i], fontsize='xx-large', color='green')

    visualizer.save(save_path)
    visualizer.close()
    print(f'Saved to {save_path}')


def vis_bev_pc_list(pc_list, name='', gts=None, save_root='./work_dirs/figs', figsize=(40, 40), color_list=None, marker_list=None, dir=None, s=0.1):
    if color_list is None:
        color_list = [
                'gray',
                'light_blue',
                'wine_red',
                'red',
                'black',
                'purple',
                'dark_green',
                'green',
        ]
    if marker_list is None:
        marker_list = ['o', ] * len(pc_list)
    if dir:
        save_root = os.path.join(save_root, dir)
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, name)
    assert '.png' in name
    if isinstance(pc_list[0], torch.Tensor):
        pc_list = [pc.cpu().detach().numpy() for pc in pc_list]
    visualizer = Visualizer2D(name='', figsize=figsize)

    if gts is not None and len(gts) > 0:
        assert isinstance(gts, LiDARInstance3DBoxes)
        gts = gts.corners.to('cpu')
        for i in range(len(gts)):
            visualizer.handler_box_4corners(gts[i, [1, 3, 7, 4], :2])

    for i, pc in enumerate(pc_list):
        if len(pc) == 0:
            continue
        visualizer.handler_pc(pc, s=s, color=color_list[i % 8], marker=marker_list[i])
    visualizer.save(save_path)
    visualizer.close()
    print(f'Saved to {save_path}')

def vis_heatmap_and_boxes(name, heatmap, corners, pc_range, resolution=(2048, 2048), dir=None, box_color='red', root=None, interpolation='bilinear', cm='jet'):

    if len(corners) == 0:
        return

    pxl_per_meter = resolution[0] / (pc_range[3] - pc_range[0])
    assert isinstance(heatmap, torch.Tensor)
    heatmap = torch.nn.functional.interpolate(heatmap[None, None, :, :], resolution, mode=interpolation)
    heatmap = heatmap.detach().cpu().numpy().astype(float)[0, 0, ...]

    corners = corners.cpu().numpy()


    assert corners.shape[1] == 8
    corners = corners[:, [1,3,7,4], :2]

    corners = np.concatenate([corners, corners[:, 0:1, :2]], axis=1)

    corners_x = ((corners[:, :, 0] - pc_range[0]) * pxl_per_meter).astype(np.int32)
    corners_y = ((corners[:, :, 1] - pc_range[1]) * pxl_per_meter).astype(np.int32)

    fig, ax = plt.subplots(figsize=(40,40))
    ax.set_xlim(0, resolution[1])
    ax.set_xlabel('Y')
    ax.set_ylim(0, resolution[0])
    ax.set_xlabel('X')
    ax.set_aspect('equal')

    ax.imshow(heatmap, cmap=cm, interpolation=interpolation)

    num_bboxes = len(corners)
    for i in range(num_bboxes):
        ax.plot(corners_x[i, :], corners_y[i, :], color=COLOR_MAP[box_color])

    if root is None:
        root = './work_dirs/figs'
    if dir:
        root = os.path.join(root, dir)
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    plt.savefig(os.path.join(root, name))
    print(f'Save to {os.path.join(root, name)}')

def vis_heatmap(name, heatmap, resolution=(2048, 2028), dir=None, box_color='red', root=None, interpolation='bilinear', cm='jet'):


    assert isinstance(heatmap, torch.Tensor)
    heatmap = torch.nn.functional.interpolate(heatmap[None, None, :, :], resolution, mode=interpolation)
    heatmap = heatmap.detach().cpu().numpy().astype(np.float)[0, 0, ...]


    fig, ax = plt.subplots(figsize=(40,40))
    ax.set_xlim(0, resolution[1])
    ax.set_xlabel('Y')
    ax.set_ylim(0, resolution[0])
    ax.set_xlabel('X')
    ax.set_aspect('equal')

    ax.imshow(heatmap, cmap=cm, interpolation=interpolation)

    if root is None:
        root = './work_dirs/figs'
    if dir:
        root = os.path.join(root, dir)
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    plt.savefig(os.path.join(root, name))
    print(f'Save to {os.path.join(root, name)}')

def vis_voting(name, pc, centers, corners=None, save_root='./work_dirs/figs', figsize=(40, 40), color_list=None, dir=None):

    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().detach().numpy()

    if isinstance(centers, torch.Tensor):
        centers = centers.cpu().detach().numpy()

    if dir:
        save_root = os.path.join(save_root, dir)
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, name)
    assert '.png' in name

    visualizer = Visualizer2D(name='', figsize=figsize)
    visualizer.handler_pc(pc, s=0.1, color='gray')
    visualizer.handler_pc(centers, s=0.5, color='red')

    if corners is not None and len(corners) > 0:
        num_gts = len(corners)
        corners = corners.cpu().numpy()
        assert corners.shape[1] == 8
        corners = corners[:, [1,3,7,4], :2]
        for i in range(num_gts):
            visualizer.handler_box_4corners(corners[i])

    visualizer.save(save_path)
    visualizer.close()
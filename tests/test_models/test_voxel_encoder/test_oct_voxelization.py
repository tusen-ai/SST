from mmdet3d.ops import OctreeVoxelization
import numpy as np, os, torch
import matplotlib.pyplot as plt


colors = np.array([
    [140, 140, 136],
    [4, 157, 217],
    [191, 4, 54],
    [0, 0, 0],
    [224, 133, 250], 
    [32, 64, 40],
    [77, 115, 67]
]) / 256


def load_pc():
    pc = np.load('pc.npy')
    pc = pc[:, :]
    # pc = pc[pc[:, 0] > 0]
    # pc = pc[pc[:, 0] < 5]
    # pc = pc[pc[:, 1] > 0]
    # pc = pc[pc[:, 1] < 5]
    return pc


def main():
    pc = torch.Tensor(load_pc()).cuda()
    print('loading complete')

    min_voxel_size = [0.32, 0.32, 8]
    max_voxel_size = [2.56, 2.56, 8]
    point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 6]
    max_num_points = 200

    voxelization = OctreeVoxelization(
        min_voxel_size, max_voxel_size, 4, point_cloud_range, max_num_points)
    print('initializing octree_voxelization')

    points, coors, oct_level = voxelization(pc)
    print('computation complete')
    coor_index = coors[:, 1] * 109 + coors[:, 2]
    pc = points.detach().cpu().numpy()
    coor_index = coor_index.detach().cpu().numpy().astype(np.int)
    pc_colors = colors[coor_index % 7]

    plt.figure(figsize=(25, 25))
    plt.axis('equal')
    plt.scatter(pc[:, 0], pc[:, 1], color=pc_colors, s=0.25)
    plt.savefig('pc.png')


if __name__ == '__main__':
    main()

import os
from os import path as osp
import av2
import torch
from ipdb import set_trace
from av2.utils.io import read_feather
import numpy as np
import multiprocessing as mp
import tqdm
import pickle as pkl

from utils import LABEL_ATTR, cuboid_to_vertices
from SO3 import quat_to_yaw

def process_single_segment(segment_path, split, info_list, ts2idx, output_dir, save_bin):
    test_mode = 'test' in split
    if not test_mode:
        segment_anno = read_feather(osp.join(segment_path, 'annotations.feather'))
    segname = segment_path.split('/')[-1]

    frame_path_list = os.listdir(osp.join(segment_path, 'sensors/lidar/'))

    for frame_name in frame_path_list:
        ts = int(osp.basename(frame_name).split('.')[0])

        if not test_mode:
            frame_anno = segment_anno[segment_anno['timestamp_ns'] == ts]
        else:
            frame_anno = None

        frame_path = osp.join(segment_path, 'sensors/lidar/', frame_name)
        frame_info = process_and_save_frame(frame_path, frame_anno, ts2idx, segname, output_dir, save_bin)
        info_list.append(frame_info)

def process_and_save_frame(frame_path, frame_anno, ts2idx, segname, output_dir, save_bin):
    frame_info = {}
    frame_info['uuid'] = segname + '/' + frame_path.split('/')[-1].split('.')[0]
    frame_info['sample_idx'] = ts2idx[frame_info['uuid']]
    frame_info['image'] = dict()
    frame_info['point_cloud'] = dict(
        num_features=4,
        velodyne_path=None,
    )
    frame_info['calib'] = dict() # not need for lidar-only
    frame_info['pose'] = dict() # not need for single frame
    frame_info['annos'] = dict(
        name=None,
        truncated=None,
        occluded=None,
        alpha=None,
        bbox=None, # not need for lidar-only
        dimensions=None,
        location=None,
        rotation_y=None,
        index=None,
        group_ids=None,
        camera_id=None,
        difficulty=None,
        num_points_in_gt=None,
    )
    frame_info['sweeps'] = [] # not need for single frame
    if frame_anno is not None:
        frame_anno = frame_anno[frame_anno['num_interior_pts'] > 0]
        cuboid_params = frame_anno.loc[:, list(LABEL_ATTR)].to_numpy()
        cuboid_params = torch.from_numpy(cuboid_params)
        yaw = quat_to_yaw(cuboid_params[:, -4:])
        xyz = cuboid_params[:, :3]
        wlh = cuboid_params[:, [4, 3, 5]]

        # waymo_yaw is equal to yaw
        # corners = cuboid_to_vertices(cuboid_params)
        # c0 = corners[:, 0, :]
        # c4 = corners[:, 4, :]
        # waymo_yaw = torch.atan2(c0[:, 1] - c4[:, 1], c0[:, 0] - c4[:, 0])
        yaw = -yaw - 0.5 * np.pi

        while (yaw < -np.pi).any():
            yaw[yaw < -np.pi] += 2 * np.pi

        while (yaw > np.pi).any():
            yaw[yaw > np.pi] -= 2 * np.pi
        
        # bbox = torch.cat([xyz, wlh, yaw.unsqueeze(1)], dim=1).numpy()
        
        cat = frame_anno['category'].to_numpy().tolist()
        cat = [c.lower().capitalize() for c in cat]
        cat = np.array(cat)

        num_obj = len(cat)

        annos = frame_info['annos']
        annos['name'] = cat
        annos['truncated'] = np.zeros(num_obj, dtype=np.float64)
        annos['occluded'] = np.zeros(num_obj, dtype=np.int64)
        annos['alpha'] = -10 * np.ones(num_obj, dtype=np.float64)
        annos['dimensions'] = wlh.numpy().astype(np.float64)
        annos['location'] = xyz.numpy().astype(np.float64)
        annos['rotation_y'] = yaw.numpy().astype(np.float64)
        annos['index'] = np.arange(num_obj, dtype=np.int32)
        annos['num_points_in_gt'] = frame_anno['num_interior_pts'].to_numpy().astype(np.int32)
    # frame_info['group_ids'] = np.arange(num_obj, dtype=np.int32)
    prefix2split = {'0': 'training', '1': 'training', '2': 'testing'}
    sample_idx = frame_info['sample_idx']
    split = prefix2split[sample_idx[0]]
    abs_save_path = osp.join(output_dir, split, 'velodyne', f'{sample_idx}.bin')
    rel_save_path = osp.join(split, 'velodyne', f'{sample_idx}.bin')
    frame_info['point_cloud']['velodyne_path'] = rel_save_path
    if save_bin:
        save_point_cloud(frame_path, abs_save_path)
    return frame_info

def save_point_cloud(frame_path, save_path):
    lidar = read_feather(frame_path)
    lidar = lidar.loc[:, ['x', 'y', 'z', 'intensity']].to_numpy().astype(np.float32)
    lidar.tofile(save_path)
    


def prepare(root):
    ts2idx = {}
    ts_list = []
    bin_idx_list = []
    seg_path_list = []
    seg_split_list = []
    assert root.split('/')[-1] == 'sensor'
    splits = ['train', 'val', 'test']
    # splits = ['train', ]
    num_train_samples = 0
    num_val_samples = 0
    num_test_samples = 0

    # 0 for training, 1 for validation and 2 for testing.
    prefixes = [0, 1, 2]

    for i in range(len(splits)):
        split = splits[i]
        prefix = prefixes[i]
        split_root = osp.join(root, split)
        seg_file_list = os.listdir(split_root)
        print(f'num of {split} segments:', len(seg_file_list))
        for seg_idx, seg_name in enumerate(seg_file_list):
            seg_path = osp.join(split_root, seg_name)
            seg_path_list.append(seg_path)
            seg_split_list.append(split)
            assert seg_idx < 1000
            frame_path_list = os.listdir(osp.join(seg_path, 'sensors/lidar/'))
            for frame_idx, frame_path in enumerate(frame_path_list):
                assert frame_idx < 1000
                bin_idx = str(prefix) + str(seg_idx).zfill(3) + str(frame_idx).zfill(3)
                ts = frame_path.split('/')[-1].split('.')[0]
                ts = seg_name + '/' + ts # ts is not unique, so add seg_name
                ts2idx[ts] = bin_idx
                ts_list.append(ts)
                bin_idx_list.append(bin_idx)
        if split == 'train':
            num_train_samples = len(ts_list)
        elif split == 'val':
            num_val_samples = len(ts_list) - num_train_samples
        else:
            num_test_samples = len(ts_list) - num_train_samples - num_val_samples
    # print three num samples
    print('num of train samples:', num_train_samples)
    print('num of val samples:', num_val_samples)
    print('num of test samples:', num_test_samples)

    assert len(ts_list) == len(set(ts_list))
    assert len(bin_idx_list) == len(set(bin_idx_list))
    return ts2idx, seg_path_list, seg_split_list

def main(seg_path_list, seg_split_list, info_list, ts2idx, output_dir, save_bin, token, num_process):
    for seg_i, seg_path in enumerate(seg_path_list):
        if seg_i % num_process != token:
            continue
        print(f'processing segment: {seg_i}/{len(seg_path_list)}')
        split = seg_split_list[seg_i]
        process_single_segment(seg_path, split, info_list, ts2idx, output_dir, save_bin)


if __name__ == '__main__':

    # please change to your data path
    root = '/mnt/weka/scratch/lve.fan/data/lidar/argo/sensor'
    output_dir = '/mnt/weka/scratch/lve.fan/FSD/data/argo2/kitti_format'
    save_bin = True
    ts2idx, seg_path_list, seg_split_list = prepare(root)

    # sample for debugging
    # seg_path_list = [s for i, s in enumerate(seg_path_list) if i % 100 == 0]
    # seg_split_list = [s for i, s in enumerate(seg_split_list) if i % 100 == 0]

    num_process = 20
    if num_process > 1:
        with mp.Manager() as manager:
            info_list = manager.list()
            pool = mp.Pool(num_process)
            for token in range(num_process):
                result = pool.apply_async(main, args=(seg_path_list, seg_split_list, info_list, ts2idx, output_dir, save_bin, token, num_process))
            pool.close()
            pool.join()
            info_list = list(info_list)
    else:
        info_list = []
        main(seg_path_list, seg_split_list, info_list, ts2idx, output_dir, save_bin, 0, 1)

    assert len(info_list) > 0
    
    train_info = [e for e in info_list if e['sample_idx'][0] == '0']
    val_info = [e for e in info_list if e['sample_idx'][0] == '1']
    test_info = [e for e in info_list if e['sample_idx'][0] == '2']
    trainval_info = train_info + val_info
    assert len(train_info) + len(val_info) + len(test_info) == len(info_list)

    # save info_list in under the output_dir as pickle file
    with open(osp.join(output_dir, 'argo2_infos_train.pkl'), 'wb') as f:
        pkl.dump(train_info, f)

    with open(osp.join(output_dir, 'argo2_infos_val.pkl'), 'wb') as f:
        pkl.dump(val_info, f)

    with open(osp.join(output_dir, 'argo2_infos_trainval.pkl'), 'wb') as f:
        pkl.dump(trainval_info, f)

    with open(osp.join(output_dir, 'argo2_infos_test.pkl'), 'wb') as f:
        pkl.dump(test_info, f)

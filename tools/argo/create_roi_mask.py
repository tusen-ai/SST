import os
from os import path as osp
from pathlib import Path
import pickle as pkl
import numpy as np
from av2.evaluation.detection.utils import load_mapped_avm_and_egoposes
from ipdb import set_trace
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
import multiprocessing as mp

def process_single_frame(info, log_to_avm, log_to_pose, output_dir, argo2_root):
    log_id, ts = info['uuid'].split('/')
    ts = int(ts)
    
    bin_path = info['point_cloud']['velodyne_path']
    bin_path = osp.join(argo2_root, 'kitti_format', bin_path)
    points = np.fromfile(bin_path, dtype=np.float32)
    points = points.reshape(-1, 4)[:, :3]

    se3 = log_to_pose[log_id][ts]
    transformed_pts = se3.transform_point_cloud(points)
    
    avm = log_to_avm[log_id]
    roi_mask = avm.get_raster_layer_points_boolean(transformed_pts, RasterLayerType.ROI)
    ground_mask = avm.get_ground_points_boolean(transformed_pts)
    drivable_mask = avm.get_raster_layer_points_boolean(transformed_pts, RasterLayerType.DRIVABLE_AREA)

    cat = np.stack([roi_mask, ground_mask, drivable_mask], axis=1)

    save_path = osp.join(output_dir, info['sample_idx'] + '.bin')
    cat.tofile(save_path)


def main(infos, log_to_avm, log_to_pose, output_dir, argo2_root, token, num_process):
    total_samples = len(infos)
    for i, info in enumerate(infos):
        if i % num_process != token:
            continue
        if i % 100 == 0:
            print(f'{i} / {total_samples}')
        process_single_frame(info, log_to_avm, log_to_pose, output_dir, argo2_root)

def prepare(infos, dataset_dir):

    log_ids = []
    for info in infos:
        log_id, ts = info['uuid'].split('/')
        ts = int(ts)
        log_ids.append(log_id)
    log_ids = list(set(log_ids))
    print(f'Got {len(log_ids)} logs')
    log_to_avm, log_to_pose = load_mapped_avm_and_egoposes(log_ids, dataset_dir)
    return log_to_avm, log_to_pose

if __name__ == '__main__':
    argo2_root = '/mnt/weka/scratch/lve.fan/SST/data/argo2'
    split = 'train'
    dataset_dir = Path(argo2_root) / 'argo2_format' / 'sensor' / split

    if split == 'test':
        kitti_split_dir = 'testing'
    else:
        kitti_split_dir = 'training'

    output_dir = osp.join(argo2_root, 'kitti_format', kitti_split_dir, 'mask')


    infos_path = f'/mnt/weka/scratch/lve.fan/transdet3d/data/argo2/kitti_format/argo2_infos_{split}.pkl'
    with open(infos_path, 'rb') as f:
        infos = pkl.load(f)

    log_to_avm, log_to_pose = prepare(infos, dataset_dir)

    num_process = 5
    if num_process > 1:
        pool = mp.Pool(num_process)
        for token in range(num_process):
            result = pool.apply_async(main, args=(infos, log_to_avm, log_to_pose, output_dir, argo2_root, token, num_process))
        pool.close()
        pool.join()
    else:
        main(infos, log_to_avm, log_to_pose, output_dir, argo2_root, 0, 1)


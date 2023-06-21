import torch
import os
import mmdet3d
from ipdb import set_trace
import tqdm
import pickle as pkl
from waymo_open_dataset.protos import metrics_pb2
from collections import defaultdict
import numpy as np

def read_bin(file_path):
    with open(file_path, 'rb') as f:
        objects = metrics_pb2.Objects()
        objects.ParseFromString(f.read())
    return objects


def waymo_object_to_mmdet(obj, version):
    '''
    According to https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto#L33
    and the definition of LiDARInstance3DBoxes
    '''
    # if gt and obj.object.num_lidar_points_in_box == 0:
    #     print('Encounter zero-point object')
    #     return None
    box = obj.object.box


    assert version < '1.0.0', 'Only support version older than 1.0.0 for now'
    heading = -box.heading - 0.5 * np.pi

    while heading < -np.pi:
        heading += 2 * np.pi
    while heading > np.pi:
        heading -= 2 * np.pi

    result = np.array(
        [
            box.center_x,
            box.center_y,
            box.center_z - box.height / 2,
            box.width,
            box.length,
            box.height,
            heading,
            obj.score,
            float(obj.object.type),
        ]
    )
    return result

def get_seed_dict(file_path, debug=False, gt=False):
    mmdet3d_version = mmdet3d.__version__
    print(f'Reading {file_path} ...')
    data = read_bin(file_path)
    objects = data.objects
    obj_dict = defaultdict(list)
    print('Collecting Bboxes ...')
    for o in tqdm.tqdm(objects):
        seg_name = o.context_name
        time_stamp = o.frame_timestamp_micros
        mm_obj = waymo_object_to_mmdet(o, mmdet3d_version)
        obj_dict[time_stamp].append(mm_obj)

    new_dict = {}

    sorted_keys = sorted(list(obj_dict.keys()))

    for k in sorted_keys:
        sample = np.stack(obj_dict[k], axis=0)
        num_obj = len(sample)
        assert num_obj > 0
        boxes = sample[:, :7]
        scores = sample[:, 7]
        gt_names = np.zeros(num_obj, dtype='<U32')
        gt_labels = np.zeros(num_obj, dtype=int)
        for name, cls_id in zip(['Car', 'Pedestrian', 'Cyclist'], [1, 2, 4]):
            this_mask = sample[:, 8] == cls_id
            gt_names[this_mask] = name
        new_dict[k] = dict(
            gt_bboxes_3d=boxes,
            scores=scores,
            gt_names=gt_names,
        )

    return new_dict

def replace_key_with_idx(input_dict, ts2idx):
    out_dict = {}
    for ts, seed in input_dict.items():
        idx = ts2idx[ts]
        out_dict[idx] = seed
    return out_dict

if __name__ == '__main__':

    # split = 'train'
    # source_model = 'fsd_sp2_ts_full6e_3f'
    split = 'test'
    source_model = 'fsd_waymoD1_1x_submission'

    assert split in ['train', 'val', 'test']

    if split == 'val':
        bin_path = f'./work_dirs/{source_model}/results.bin'
    else:
        bin_path = f'./work_dirs/{source_model}/results_{split}.bin'

    data_root = 'data/waymo/kitti_format/'
    save_path = os.path.join(data_root, f'{source_model}_seed_prediction_{split}.pkl')
    assert not os.path.exists(save_path)

    with open('./data/idx2timestamp.pkl', 'rb') as fr: 
        idx2ts = pkl.load(fr)          

    ts2idx = {}
    for idx, ts in idx2ts.items():
        ts2idx[ts] = idx

    seed_dict = get_seed_dict(bin_path)
    seed_dict = replace_key_with_idx(seed_dict, ts2idx)

    with open(save_path, 'wb') as fw:
        pkl.dump(seed_dict, fw)
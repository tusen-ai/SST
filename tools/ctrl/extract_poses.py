import os, json
from collections import defaultdict
import numpy as np
import pickle as pkl

from ipdb import set_trace
from tqdm import tqdm
    
if __name__ == '__main__':
    splits = ['training', 'validation', 'testing']
    info_path_list = [
        './data/waymo/kitti_format/waymo_infos_train.pkl',
        './data/waymo/kitti_format/waymo_infos_val.pkl',
        './data/waymo/kitti_format/waymo_infos_test.pkl'
    ]

    with open('./data/waymo/kitti_format/idx2timestamp.pkl', 'rb') as fr:
        idx2timestamp = pkl.load(fr)

    with open('./data/waymo/kitti_format/idx2contextname.pkl', 'rb') as fr:
        idx2contextname = pkl.load(fr)
    
    pose_dict = {}
    context2ts = defaultdict(list)

    for path in info_path_list:
        with open(path, 'rb') as fr:
            mmdet_info = pkl.load(fr)
        
        for info in mmdet_info:
            idx_str = info['point_cloud']['velodyne_path'].split('/')[-1].split('.')[0]
            pose = info['pose']
            ts = idx2timestamp[idx_str]
            context = idx2contextname[idx_str]
            context2ts[context].append(ts)
            pose_dict[ts] = pose
    
    for k, v in context2ts.items():
        context2ts[k] = sorted(v)
    
    assert sum([len(v) for _, v in context2ts.items()]) == len(idx2timestamp)
    assert len(pose_dict) == len(idx2timestamp)
    
    with open('./data/waymo/kitti_format/context2timestamp.pkl', 'wb') as fw:
        pkl.dump(context2ts, fw)

    with open('./data/waymo/kitti_format/poses.pkl', 'wb') as fw:
        pkl.dump(pose_dict, fw)
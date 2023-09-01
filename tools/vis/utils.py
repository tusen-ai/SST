import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict

from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.protos import metrics_pb2

from ipdb import set_trace
from visualizer import BBox

import torch

def read_bin(file_path):
    with open(file_path, 'rb') as f:
        objects = metrics_pb2.Objects()
        objects.ParseFromString(f.read())
    return objects

def object2array(obj):
    """transform box dict in waymo_open_format to array
    Args:
        box_dict ([dict]): waymo_open_dataset formatted bbox
    """
    box = obj.object.box
    result = np.array([
        box.center_x,
        box.center_y,
        box.center_z,
        box.width,
        box.length,
        box.height,
        box.heading,
        obj.score,
        float(obj.object.type),
    ])
    return result

def object2BBox(obj):
    box = obj.object.box
    result = BBox(
        box.center_x,
        box.center_y,
        box.center_z,
        box.height,
        box.width,
        box.length,
        box.heading,
        obj.score,
        float(obj.object.type),
    )
    return result

def object2mmdetformat(obj):
    '''
    According to https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto#L33
    and the definition of LiDARInstance3DBoxes
    '''
    box = obj.object.box
    heading = box.heading - 1.5 * np.pi

    while heading < -np.pi:
        heading += 2 * np.pi
    while heading > np.pi:
        heading -= 2 * np.pi

    result = np.array(
        [
            box.center_x,
            box.center_y,
            box.center_z,
            box.width,
            box.length,
            box.height,
            heading,
            obj.score,
            float(obj.object.type),
        ]
    )
    return result

def get_obj_dict_from_bin_file(file_path, debug=False, to_mmdet=False, concat=False):
    print(f'Reading {file_path} ...')
    pred_data = read_bin(file_path)
    objects = pred_data.objects
    if debug:
        objects = objects[:10]
    obj_dict = defaultdict(list)
    print('Collecting Bboxes ...')
    for o in tqdm(objects):
        seg_name = o.context_name
        time_stamp = o.frame_timestamp_micros
        if to_mmdet:
            bbox_for_vis = object2mmdetformat(o)
        else:
            bbox_for_vis = object2BBox(o)
        obj_dict[time_stamp].append(bbox_for_vis)
    
    if concat:
        for k in obj_dict:
            obj_list = obj_dict[k]
            obj_dict[k] = np.stack(obj_list, dim=0)
    
    return obj_dict

idx2ts = None

def get_pc_from_time_stamp(timestamp, path, split='training'):
    global idx2ts
    if idx2ts is None:
        with open(path, 'rb') as fr:
            idx2ts = pkl.load(fr)
        print('Read idx2ts')
    ts2idx = {}
    for idx, ts in idx2ts.items():
        ts2idx[ts] = idx
    
    curr_idx = ts2idx[timestamp]
    pc_root = f'./data/waymo/kitti_format/{split}/velodyne'
    pc_path = os.path.join(pc_root, curr_idx + '.bin')
    pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 6)
    pc = pc[:, :3]
    return pc

from argo2mmdet import prepare
from ipdb import set_trace
import os
from os import path as osp
from av2.utils.io import read_feather
import pandas as pd

if __name__ == '__main__':
    root = '/mnt/weka/scratch/lve.fan/data/lidar/argo/sensor'
    save_path = '/mnt/weka/scratch/lve.fan/data/lidar/argo/sensor/val_anno.feather'
    ts2idx, seg_path_list, seg_split_list = prepare(root)
    val_seg_path_list = [seg_path for seg_path in seg_path_list if 'val' in seg_path]
    assert len(val_seg_path_list) == len([i for i in seg_split_list if i == 'val'])

    seg_anno_list = []
    for seg_path in val_seg_path_list:
        seg_anno = read_feather(osp.join(seg_path, 'annotations.feather'))
        log_id = seg_path.split('/')[-1]
        seg_anno["log_id"] = log_id
        seg_anno_list.append(seg_anno)
    
    gts = pd.concat(seg_anno_list).reset_index()
    gts.to_feather(save_path)

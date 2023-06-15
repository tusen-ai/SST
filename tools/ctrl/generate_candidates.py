import torch.multiprocessing as mp
import os, yaml, pickle as pkl, argparse
from os import path as osp
from utils import read_bin, generate_tracklets, get_pc_from_time_stamp
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from ipdb import set_trace
from mmdet3d.core import LiDARTracklet
import time

def tracklet_list2dict(tracklets):
    trks_dict = defaultdict(list)
    for trk in tracklets:
        segname = trk.segment_name
        trks_dict[segname].append(trk)
    return trks_dict

def tracklet_assign_wrapper(tracklet_pd_dict, tracklet_gt_dict, config, process):
    print('Begin assign')
    manager = mp.Manager()
    out_dict = manager.dict()
    counter_list = manager.list()
    keys_list = list(tracklet_pd_dict.keys())
    if process > 1:
        pool = mp.Pool(process)
        for token in range(process):
            result = pool.apply_async(
                tracklet_assign,
                args=(tracklet_pd_dict, tracklet_gt_dict, keys_list, out_dict, counter_list, config, token, process)
            )
        pool.close()
        pool.join()
    else:
        tracklet_assign(tracklet_pd_dict, tracklet_gt_dict, keys_list, out_dict, counter_list, config)
    return out_dict

def tracklet_assign(tracklet_pd_dict, tracklet_gt_dict, keys_list, out_dict, counter_list, config, token=0, process=1):
    try:
        print('Enter workers')
        torch.cuda.set_device(token % 8)
        for i, segname in enumerate(keys_list):

            if i % process != token:
                continue

            trks_pd = tracklet_pd_dict[segname]
            for t in trks_pd:
                t.from_collate_format()

            if segname not in tracklet_gt_dict:
                counter_list.append(segname)
                print(f'Finish {len(counter_list)}/{len(keys_list)}')
                continue

            trks_gt = tracklet_gt_dict[segname]
            for t in trks_gt:
                t.from_collate_format()

            for t_pd in trks_pd:
                affinity = [t_pd.max_iou(t_gt) for t_gt in trks_gt]
                candidates_this_pd = [trks_gt[i] for i in range(len(trks_gt)) if affinity[i] > config['candidate']['affinity_thresh']]
                candidates_this_pd = [e.to_dump_format() for e in candidates_this_pd]
                out_dict[t_pd.uuid] = candidates_this_pd
            counter_list.append(segname)
            print(f'Finish {len(counter_list)}/{len(keys_list)}')
    except Exception as e:
        print(e)

def stats(tracklets, candidates_list):
    num_pred_boxes = sum([len(t) for t in tracklets])
    num_pred_trks = len(tracklets)
    unmatched_trks = [tracklets[i] for i in range(len(tracklets)) if len(candidates_list[i]) == 0]
    num_unmatched_boxes = sum([len(t) for t in unmatched_trks])
    print(f'Tracklet FP rate: {len(unmatched_trks) / num_pred_trks}')
    print(f'Box FP rate: {num_unmatched_boxes / num_pred_boxes}')


parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--gt-bin-path', type=str, default='./data/waymo/waymo_format/train_gt.bin')
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    beg = time.time()

    config = yaml.load(open(args.config, 'r'))
    config_name = os.path.basename(args.config).split('.')[0]
    data_root = config['data_root']
    split = config['split']
    assert split in ('training', 'val', 'test')

    save_path = osp.join(data_root, config_name + f'_{split}' + '_gt_candidates.pkl')
    print(f'Results will be saved to {save_path}')

    gt_bin_path = args.gt_bin_path
    if split == 'val':
        gt_bin_path = gt_bin_path.replace('train_gt.bin', 'gt.bin')
    gt_bin_data = read_bin(gt_bin_path)
    types = set(config['type'])
    tracklets_gt = generate_tracklets(gt_bin_data, types)

    for t in tracklets_gt:
        t.to_collate_format()

    print('Loading pickle infos...')

    info_path = osp.join(data_root, config_name + f'_{split}' + '.pkl')
    with open(info_path, 'rb') as fr:
        tracklets_pd = pkl.load(fr)

    print('Recover from dumped format...')

    tracklets_pd = [LiDARTracklet.from_dump_format(e) for e in tqdm(tracklets_pd)]

    for t in tracklets_pd:
        t.to_collate_format()

    trks_gt_dict = tracklet_list2dict(tracklets_gt)
    trks_pd_dict = tracklet_list2dict(tracklets_pd)

    candidates_of_each_trk = tracklet_assign_wrapper(trks_pd_dict, trks_gt_dict, config, args.process) # out as dumped format

    num_candidates = sum([len(v) for _, v in candidates_of_each_trk.items()])
    print(f'Average candidates per trk {num_candidates/len(candidates_of_each_trk)}')

    list_to_save = [candidates_of_each_trk.get(t.uuid, []) for t in tracklets_pd]

    stats(tracklets_pd, list_to_save)

    with open(save_path, 'wb') as fw:
        pkl.dump(list_to_save, fw)
    end = time.time()
    print(f'Time cost: {end - beg} seconds.')
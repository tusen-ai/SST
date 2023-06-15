import os, yaml, pickle as pkl, argparse
from os import path as osp
from utils import read_bin, generate_tracklets, get_pc_from_time_stamp
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from ipdb import set_trace
import multiprocessing
import time

def select_tracklets(config, tracklets):
    mode = config['selection']['mode']
    size = config['selection']['size']
    if mode == 'random':
        interval = int(1 / size)
        return tracklets[::interval]
    raise NotImplementedError

def extract_and_save_point_wrapper(config, tracklets, timestamps, timestamps2idx, mm_data_root, save_dir, process):
    manager = multiprocessing.Manager()
    os.makedirs(save_dir, exist_ok=config['exist_ok']) # in case of overwrite
    # trks_dict = manager.dict()
    num_pts_dict = manager.dict()

    # for trk in tracklets:
    #     trk.to_collate_format()
    #     segname = trk.segment_name
    #     if segname not in trks_dict:
    #         trks_dict[segname] = manager.list()
    #         trks_dict[segname].append(trk)

    trks_dict = defaultdict(list)
    for trk in tracklets:
        trk.to_collate_format()
        segname = trk.segment_name
        trks_dict[segname].append(trk)

    keys_list = list(trks_dict.keys())

    counter_list = manager.list()
    beg = time.time()

    if process > 1:
        pool = multiprocessing.Pool(process)
        for token in range(process):
            result = pool.apply_async(
                extract_and_save_point,
                args=(config, trks_dict, num_pts_dict, keys_list, timestamps, timestamps2idx, mm_data_root, save_dir, counter_list, token, process)
            )
        pool.close()
        pool.join()
    else:
        extract_and_save_point(config, trks_dict, num_pts_dict, keys_list, timestamps, timestamps2idx, mm_data_root, save_dir, counter_list)

    print('Assign num points...')

    for segname in trks_dict:
        trks = trks_dict[segname]
        num_pts_list = num_pts_dict[segname]
        assert len(trks) == len(num_pts_list)
        for trk, n in zip(trks, num_pts_list):
            trk.num_pts_in_boxes = n

    end = time.time()
    print(f'Extract time cost: {end - beg}s')


def extract_and_save_point(config, tracklets_dict, num_pts_dict, dict_keys, timestamps, timestamps2idx, mm_data_root, save_dir, counter_list, token=0, process=1):
    try:
        torch.cuda.set_device(token % 8)
        if config['split'] in ('training', 'val'):
            kitti_split = 'training'
        else:
            kitti_split = 'testing'

        for seg_idx, segname in enumerate(dict_keys):

            if seg_idx % process != token:
                continue

            trks_this_seg = tracklets_dict[segname]
            for trk in trks_this_seg:
                trk.from_collate_format()

            full_ts = timestamps[segname]

            for ts in full_ts:
                boxes_this_ts = [trk[ts] for trk in trks_this_seg]
                pc = get_pc_from_time_stamp(ts, timestamps2idx, mm_data_root, kitti_split)
                pc = torch.from_numpy(pc).cuda()

                for i, box in enumerate(boxes_this_ts):
                    if box is None:
                        continue
                    box = box.enlarged_box(config['box']['extra_width'])
                    box.tensor = box.tensor.cuda()
                    inbox_inds = box.points_in_boxes(pc[:, :3])
                    assert (inbox_inds <= 0).all()
                    inbox_pc = pc[inbox_inds == 0]
                    trks_this_seg[i].append_pc(inbox_pc.cpu().numpy(), ts)

            num_pts_list = []
            for trk in trks_this_seg:
                pc = trk.pc_list

                num_pts_in_boxes = [len(p) for p in pc]

                filename = osp.join(save_dir, trk.segment_name + '--' + trk.id + '.npy')
                with open(filename, 'wb') as fw:
                    np.save(fw, pc)
                num_pts_list.append(num_pts_in_boxes)
                trk.free_pc()
            num_pts_dict[segname] = num_pts_list
            counter_list.append(seg_idx)
            print(f'Finish {len(counter_list)}/{len(dict_keys)}')
    except Exception as e:
        print("Got Error ", e)

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()

if __name__ == '__main__':
    # assert args.process == 1, 'Mulitprocess cannot work. Cannot pass tracklets_dict into target function, I dont know why'
    multiprocessing.set_start_method('spawn', force=True)

    config = yaml.load(open(args.config, 'r'))
    config_name = os.path.basename(args.config).split('.')[0]
    data_root = config['data_root']

    split = config['split']
    assert split in ('training', 'val', 'test')
    save_dir = osp.join(data_root, config_name + f'_{split}' + '_database')
    info_path = osp.join(data_root, config_name + f'_{split}' + '.pkl')
    if not config['exist_ok']:
        assert not os.path.exists(info_path)
    print(f'Point clouds will be saved to {save_dir}')
    print(f'Pickled Info will be saved to {info_path}')

    if split == 'val':
        bin_data = read_bin(config['val_bin_path'])
    elif split == 'test':
        bin_data = read_bin(config['test_bin_path'])
    else:
        bin_data = read_bin(config['bin_path'])

    tracklets = generate_tracklets(bin_data)
    if split == 'training':
        tracklets = select_tracklets(config, tracklets)

    mm_data_root = './data/waymo/kitti_format'
    with open(osp.join(mm_data_root, 'idx2timestamp.pkl'), 'rb') as fr:
        idx2ts = pkl.load(fr)
    timestamps2idx = {ts:idx for idx, ts in idx2ts.items()}

    with open(osp.join(mm_data_root, 'idx2contextname.pkl'), 'rb') as fr:
        idx2segname = pkl.load(fr)

    ts2segname = {idx2ts[idx]:segname for idx, segname in idx2segname.items()}

    timestamps_all_seg = defaultdict(list)
    for ts, segname in ts2segname.items():
        timestamps_all_seg[segname].append(ts)

    for segname, ts_list in timestamps_all_seg.items():
        ts_list.sort()

    extract_and_save_point_wrapper(config, tracklets, timestamps_all_seg, timestamps2idx, mm_data_root, save_dir, args.process)

    for t in tracklets:
        assert t.num_pts_in_boxes is not None

    print('Convert to dump format')
    dump_format = [t.to_dump_format() for t in tracklets]
    with open(info_path, 'wb') as fw:
        pkl.dump(dump_format, fw)

    print(f'Tracklets saved to {info_path}')
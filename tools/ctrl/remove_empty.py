from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from tqdm import tqdm
from ipdb import set_trace
import multiprocessing
import sys
sys.path.append('.')
import numpy as np
import os
from os import path as osp
import argparse, yaml
from utils import bin2lidarboxes, get_pc_from_time_stamp
import pickle as pkl
import torch

def read_bin(file_path):
    with open(file_path, 'rb') as f:
        objects = metrics_pb2.Objects()
        objects.ParseFromString(f.read())
    return objects


def save_to_bin(out_list, save_path, prefix=None):
    objects = metrics_pb2.Objects()
    for out in out_list:
        boxes, ids, segname, ts = out
        assert isinstance(ts, int)

        for i in range(len(boxes)):
            box_np, id_ = boxes[i], ids[i].item()

            o = metrics_pb2.Object()
            o.context_name = segname
            o.frame_timestamp_micros = ts
            box = label_pb2.Label.Box()

            box.center_x, box.center_y, box.center_z, box.width, box.length, box.height, box.heading, o.score, type_ = box_np.tolist()
            o.object.box.CopyFrom(box)

            o.object.id = id_
            o.object.type = int(type_)
            objects.objects.append(o)

    if prefix is not None:
        dir_path = osp.dirname(save_path)
        save_path = osp.join(dir_path, prefix + '_' + osp.basename(save_path))
    print(f'Num objects after remove: {len(objects.objects)}')
    f = open(save_path, 'wb')
    f.write(objects.SerializeToString())
    f.close()


def call_bin(save_path):
    import subprocess
    print('Start evaluating bin file...')
    ret_bytes = subprocess.check_output(
        f'./mmdet3d/core/evaluation/waymo_utils/compute_detection_metrics_main {save_path} ' + './data/waymo/waymo_format/gt.bin', shell=True)
    ret_texts = ret_bytes.decode('utf-8')
    print(ret_texts)
    txt_path = save_path.replace('.bin', '.txt')
    with open(txt_path, 'w') as fw:
        fw.write(ret_texts)

def process_single_frame(config, data_root, out_list, input_list, ts2idx, kitti_split, token, num_process):
    print(f'Process {token} starts...')

    try:
        torch.cuda.set_device(token % 8)

        from mmdet3d.core import LiDARInstance3DBoxes

        num_frames = len(input_list)
        for frame_idx in range(num_frames):
            if frame_idx % num_process != token:
                continue

            cur_len = len(out_list)
            if cur_len % 1000 == 0:
                print(f'{cur_len} / {num_frames}')

            ori_boxes, boxes, ids, segname, ts = input_list[frame_idx]
            boxes = LiDARInstance3DBoxes(boxes, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))

            pc = get_pc_from_time_stamp(ts, ts2idx, data_root, split=kitti_split)
            pc = torch.from_numpy(pc[:, :3]).cuda()
            boxes.tensor = boxes.tensor.cuda()
            boxes.tensor[:, 2] += boxes.tensor[:, 5] * config['bottom_lift']
            if config.get('extra_hw', None) is not None:
                boxes = boxes.enlarged_box_hw(config['extra_hw'])
            if not config.get('box_grouping', False):
                inbox_inds = boxes.points_in_boxes(pc)
                inbox_inds = inbox_inds[inbox_inds > -1]
                if len(inbox_inds) == 0:
                    continue
                valid_mask = torch.zeros(len(boxes), dtype=torch.bool, device=inbox_inds.device)
                unq_inds = torch.unique(inbox_inds)
                assert unq_inds.max().item() < len(boxes)
                valid_mask[unq_inds.long()] = True

                valid_mask_np = valid_mask.cpu().numpy()
            else:
                valid_mask_np = get_grouping_mask(boxes, pc, config)

            valid_box = ori_boxes[valid_mask_np]
            valid_ids = ids[valid_mask_np]

            frame_results = (valid_box, valid_ids, segname, ts)

            out_list.append(frame_results)

    except Exception as e:
        print(e)

def get_grouping_mask(boxes, pc, config):
    group_size = config['group_size']
    shuffle_inds = torch.randperm(len(boxes), dtype=torch.long, device=pc.device)
    shuffle_inds_list = shuffle_inds.split(group_size)
    valid_mask = torch.zeros(len(boxes), dtype=torch.bool, device=pc.device)

    for inds in shuffle_inds_list:
        this_boxes = boxes[inds]
        this_valid_mask = torch.zeros(len(this_boxes), dtype=torch.bool, device=pc.device)
        inbox_inds = this_boxes.points_in_boxes(pc)
        inbox_inds = inbox_inds[inbox_inds > -1]
        if len(inbox_inds) == 0:
            continue
        unq_inds = torch.unique(inbox_inds).long()
        assert unq_inds.max().item() < len(boxes)
        this_valid_mask[unq_inds] = True
        valid_mask[inds] = this_valid_mask

    out_mask = valid_mask.cpu().numpy()
    return out_mask





parser = argparse.ArgumentParser()
# parser.add_argument('config', type=str)
parser.add_argument('--bin-path', type=str, default='./mot_results/waymo/validation/immortal_gpu_real3d_fsdpp/bin/pred_inter.bin')
parser.add_argument('--split', type=str, default='validation')
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--type', type=str, default='vehicle')
args = parser.parse_args()

if __name__ == '__main__':
    bin_path = args.bin_path
    old_bin_name = osp.basename(bin_path).split('.')[0]
    assert args.split in ('training', 'testing') # training for train set and validation set; testing for test set.

    # suffix = osp.basename(args.config).split('.')[0]
    # config = yaml.load(open(args.config, 'r'))
    if args.type == 'vehicle':
        config = {'bottom_lift': 0.2, 'extra_hw':0.0, 'box_grouping':True, 'group_size':1}
    elif args.type == 'pedestrian':
        config = {'bottom_lift': 0.1, 'extra_hw':0.0, 'box_grouping':True, 'group_size':1}
    elif args.type == 'cyclist':
        config = {'bottom_lift': 0.1, 'extra_hw':0.0, 'box_grouping':True, 'group_size':1}
    else:
        raise NotImplementedError


    save_path = osp.join(osp.dirname(bin_path), old_bin_name + f'_wo_empty_right.bin')
    print(f'Results will be saved to {save_path}')

    waymo_data_root = './data/waymo/kitti_format'
    idx2ts_path = osp.join(waymo_data_root, 'idx2timestamp.pkl')
    with open(idx2ts_path, 'rb') as fr:
        idx2ts = pkl.load(fr)

    ts2idx = {ts:idx for idx, ts in idx2ts.items()}

    bin_data = read_bin(bin_path)
    input_list = bin2lidarboxes(bin_data, )

    manager = multiprocessing.Manager()
    out_list = manager.list()

    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            pool.apply_async(process_single_frame, args=(config, waymo_data_root, out_list, input_list, ts2idx, args.split, token, args.process))
        pool.close()
        pool.join()
    else:
        process_single_frame(config, waymo_data_root, out_list, input_list, args.split, ts2idx, 0, 1)

    print('Convert results to bin file...')
    save_to_bin(out_list, save_path)
    if args.split != 'testing':
        call_bin(save_path)
import os, yaml, pickle as pkl, argparse
from os import path as osp
from utils import read_bin, generate_tracklets, get_pc_from_time_stamp
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from ipdb import set_trace
import time
# from mmdet3d.utils import vis_bev_pc

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

pi = 3.1415926

num_turns = 0
num_all_cnt = 0

def check_track(track):

    global num_turns
    global num_all_cnt

    if len(track) < 3:
        return False

    boxes = track.concated_boxes()
    yaws = boxes.tensor[:, -1]
    yaws1 = yaws[:-1]
    yaws2 = yaws[1:]
    diff = yaws1 - yaws2
    low_mask = diff < -pi
    if low_mask.any():
        diff[low_mask] += 2*pi

    high_mask = diff > pi
    if high_mask.any():
        diff[high_mask] -= 2*pi

    this_num_turns = (diff.abs() > 3*pi/4).sum()
    num_turns += this_num_turns
    num_all_cnt += len(diff)
    if this_num_turns > 0:
        return True

    # vol = boxes.volume
    # vol1 = vol[:-1]
    # vol2 = vol[1:]

    # diff = (vol1 - vol2).abs() / vol1
    # if (diff > 0.2).any():
    #     return True

    return False

def lidar2waymo_box(in_box, score, obj_type, context_name, timestamp):

    box = label_pb2.Label.Box()
    height = in_box[5].item()
    heading = in_box[6].item()

    heading = -heading - 0.5 * 3.1415926

    while heading < -3.141593: 
        heading += 2 * 3.141592
    while heading >  3.141593:
        heading -= 2 * 3.141592

    box.center_x = in_box[0].item()
    box.center_y = in_box[1].item()
    box.center_z = in_box[2].item() + height / 2
    box.length = in_box[4].item()
    box.width = in_box[3].item()
    box.height = height
    box.heading = heading

    o = metrics_pb2.Object()
    o.object.box.CopyFrom(box)
    o.object.type = obj_type
    o.score = score

    o.context_name = context_name
    o.frame_timestamp_micros = timestamp

    return o

def convert_tracklet_to_waymo(tracklets, save_path):
    import tqdm

    bin_file = metrics_pb2.Objects()

    print('\nStarting convert to waymo ...')
    for trk in tqdm.tqdm(tracklets):
        trk_id = trk.id
        assert trk.type in (1, 2, 4), 'For waymo'
        assert isinstance(trk_id, str)

        for i in range(len(trk)):
            o = lidar2waymo_box(
                trk.box_list[i].tensor.numpy().squeeze(),
                trk.score_list[i],
                trk.type,
                trk.segment_name,
                trk.ts_list[i],
            )
            o.object.id = trk_id
            bin_file.objects.append(o)
    num_obj = len(bin_file.objects)

    f = open(save_path, 'wb')
    f.write(bin_file.SerializeToString())
    f.close()
    print(f'Convert finished, got {num_obj} objects. Saved to {save_path}')

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


parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
args = parser.parse_args()

if __name__ == '__main__':


    config = yaml.load(open(args.config, 'r'))
    config_name = os.path.basename(args.config).split('.')[0]
    bin_path = config['bin_path']
    bin_data = read_bin(bin_path)
    save_path = bin_path.replace('.bin', f'_{config_name}.bin')
    print(f'Result will be saved to {save_path}')

    with open('./data/waymo/kitti_format/poses.pkl', 'rb') as fr:
        ts2pose = pkl.load(fr)

    with open('./data/waymo/kitti_format/context2timestamp.pkl', 'rb') as fr:
        segname2ts = pkl.load(fr)

    for ts, pose in ts2pose.items():
        ts2pose[ts] = torch.from_numpy(pose).float()

    print(f'Got {len(bin_data.objects)} objects before extending')
    tracklets = generate_tracklets(bin_data)#[::100]

    print('Set pose and transform...')
    for trk in tqdm(tracklets):
        trk.set_poses(ts2pose)
        trk.frame_transform(trk.pose_list[0])

    print('Extending ...')
    extend_all_ts = config.get('extend_all', False)
    print(f'Extend all timestamps? {extend_all_ts}')

    for i, trk in tqdm(enumerate(tracklets)):
        trk.set_velocity()
        # trk.set_acceleration()
        full_ts_list = segname2ts[trk.segment_name]
        # bboxes = trk.concated_boxes()
        # vis_bev_pc(pc=None, gts=bboxes, name=f'{i}_before_ext_{len(bboxes)}.png', dir='track_extension', figsize=(20, 20))

        if extend_all_ts and len(trk) > config['min_length_to_extend_all']:
            trk.extend_all(
                full_ts_list,
                config['min_length_to_extend_all'],
                ts2pose,
                config['score_multiplier'],
                config['velo_window_size']
            )
        else:
            trk.extend(
                config['extend_length'],
                config['direction'],
                full_ts_list,
                config['min_length_to_extend'],
                ts2pose,
                config['score_multiplier'],
                config['velo_window_size']
            )

        trk.shared2ego(inplace=True)
        # bboxes = trk.concated_boxes()
        # vis_bev_pc(pc=None, gts=bboxes, name=f'{i}_after_ext_{len(bboxes)}.png', dir='track_extension', figsize=(20, 20))

    convert_tracklet_to_waymo(tracklets, save_path)
    call_bin(save_path)
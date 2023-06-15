import numpy as np
from ipdb import set_trace
from tqdm import tqdm
import os
from os import path as osp

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


def read_bin(file_path):
    with open(file_path, 'rb') as f:
        objects = metrics_pb2.Objects()
        objects.ParseFromString(f.read())
    return objects

def generate_tracklets(bin_data, types=None):
    from mmdet3d.core import LiDARInstance3DBoxes, LiDARTracklet
    tracklets = {}
    objects = bin_data.objects
    if types is None:
        types = (1, 2, 4)
    for i, o in tqdm(enumerate(objects)):

        cat = o.object.type
        # only keep vehicle, pedestrian, cyclist
        if cat not in types:
            continue
        obj_id = o.object.id
        ts = o.frame_timestamp_micros
        seg_name = o.context_name
        obj_uuid = seg_name + '-' + obj_id
        box = o.object.box

        heading = -box.heading - 0.5 * np.pi
        while heading < -np.pi:
            heading += 2 * np.pi
        while heading > np.pi:
            heading -= 2 * np.pi

        box_np = np.array([[box.center_x, box.center_y, box.center_z - box.height / 2, box.width, box.length, box.height, heading]], dtype=np.float32)
        box_lidar = LiDARInstance3DBoxes(box_np, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0))
        # box_lidar = box_np
        score = o.score
        if obj_uuid not in tracklets:
            new_tracklet = LiDARTracklet(seg_name, obj_id, cat, False)
            new_tracklet.append(box_lidar, score, ts, False)
            tracklets[obj_uuid] = new_tracklet
        else:
            tracklets[obj_uuid].append(box_lidar, score, ts, False)

    tracklet_list = []
    for _, trk in tracklets.items():
        trk.freeze()
        tracklet_list.append(trk)

    return tracklet_list

def get_pc_from_time_stamp(timestamp, ts2idx, data_root, split='training'):

    curr_idx = ts2idx[timestamp]
    pc_root = osp.join(data_root, f'{split}/velodyne')
    pc_path = os.path.join(pc_root, curr_idx + '.bin')
    pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 6)
    return pc

def waymo_object_to_mmdet(obj, version):
    '''
    According to https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto#L33
    '''
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
            box.center_z,
            box.width,
            box.length,
            box.height,
            heading,
            # obj.score,
            # float(obj.object.type),
        ]
    )
    return result

def waymo_object_to_array(obj):
    box = obj.object.box
    result = np.array(
        [
            box.center_x,
            box.center_y,
            box.center_z,
            box.width,
            box.length,
            box.height,
            box.heading,
            obj.score,
            float(obj.object.type),
        ]
    )
    return result

from collections import defaultdict
def bin2lidarboxes(bin_data, debug=False, gt=False):
    objects = bin_data.objects
    obj_dict = defaultdict(list)
    ori_obj_dict = defaultdict(list)
    id_dict = defaultdict(list)
    segname_dict = {}
    print('Collecting Bboxes ...')
    for o in tqdm(objects):
        seg_name = o.context_name
        time_stamp = o.frame_timestamp_micros
        obj_id = o.object.id
        mm_obj = waymo_object_to_mmdet(o, '0.15.0')
        if mm_obj is not None:
            obj_dict[time_stamp].append(mm_obj)
            ori_obj_dict[time_stamp].append(waymo_object_to_array(o))
            id_dict[time_stamp].append(obj_id)
            segname_dict[time_stamp] = seg_name

    out_list = []

    for ts in tqdm(obj_dict):

        boxes = np.stack(obj_dict[ts], axis=0)[:, :7]
        ori_boxes = np.stack(ori_obj_dict[ts])
        ids = np.stack(id_dict[ts])

        e = (ori_boxes, boxes, ids, segname_dict[ts], ts)
        out_list.append(e)

    return out_list

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
        assert trk.type == 1, 'For now'
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

    f = open(save_path, 'wb')
    f.write(bin_file.SerializeToString())
    f.close()
    print(f'Convert finished. Saved to {save_path}')

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
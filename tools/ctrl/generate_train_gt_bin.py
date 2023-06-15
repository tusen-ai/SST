import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import argparse
import json
from google.protobuf.descriptor import FieldDescriptor as FD
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import metrics_pb2

from tqdm import tqdm
from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, default='./data/waymo/waymo_format/training')
parser.add_argument('--output', type=str, default='./data/waymo/waymo_format/train_gt.bin',
    help='the location of output information')
args = parser.parse_args()

def extract_objects_from_frame(frame, context_name):
    out_list = []
    for label in frame.laser_labels:
        o = metrics_pb2.Object()
        o.context_name = context_name
        o.frame_timestamp_micros = frame.timestamp_micros
        o.object.box.CopyFrom(label.box)
        o.object.id = label.id
        o.object.type = label.type
        o.object.metadata.CopyFrom(label.metadata)
        o.object.num_lidar_points_in_box = label.num_lidar_points_in_box
        o.object.detection_difficulty_level = label.detection_difficulty_level 
        o.object.tracking_difficulty_level = label.tracking_difficulty_level 
        out_list.append(o)
    return out_list


def main(data_folder, save_path):
    print(f'Result will be saved to {save_path}')
    tf_records = os.listdir(data_folder)
    tf_records = [x for x in tf_records if 'tfrecord' in x]
    tf_records = sorted(tf_records)

    bin_data = metrics_pb2.Objects()

    for record_index, tf_record_name in tqdm(enumerate(tf_records)):
        FILE_NAME = os.path.join(data_folder, tf_record_name)
        dataset = tf.data.TFRecordDataset(FILE_NAME, compression_type='')

        context_name = tf_record_name.replace('segment-', '').replace('_with_camera_labels.tfrecord', '')
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            obj_list = extract_objects_from_frame(frame, context_name)
            for obj in obj_list:
                bin_data.objects.append(obj)

    f = open(save_path, 'wb')
    f.write(bin_data.SerializeToString())
    f.close()


if __name__ == '__main__':
    main(args.data_folder, args.output)
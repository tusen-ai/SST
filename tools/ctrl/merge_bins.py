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



if __name__ == '__main__':
    bin_path_list = []
    out_path = 'xxx.bin'

    assert not os.path.exists(out_path)
    bin_data_list = [read_bin(p) for p in bin_path_list]

    out_bin = metrics_pb2.Objects()

    for bin_data in bin_data_list:
        out_bin.objects.extend(bin_data.objects)

    f = open(out_path, 'wb')
    f.write(out_bin.SerializeToString())
    f.close()
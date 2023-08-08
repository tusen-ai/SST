import numpy as np
import pickle as pkl
from ipdb import set_trace
from collections import defaultdict
from tqdm import tqdm

class_names = ['Regular_vehicle',

 'Pedestrian',
 'Bicyclist',
 'Motorcyclist',
 'Wheeled_rider',

 'Bollard',
 'Construction_cone',
 'Sign',
 'Construction_barrel',
 'Stop_sign',
 'Mobile_pedestrian_crossing_sign',

 'Large_vehicle',
 'Bus',
 'Box_truck',
 'Truck',
 'Vehicular_trailer',
 'Truck_cab',
 'School_bus',
 'Articulated_bus',
 'Message_board_trailer',

 'Bicycle',
 'Motorcycle',
 'Wheeled_device',
 'Wheelchair',
 'Stroller',

 'Dog']

def get_average_sizes(infos):
    cls_dict = defaultdict(list)
    for e in tqdm(infos):
        names = e['annos']['name']
        unq_names = np.unique(names)
        for n in unq_names:
            n = n.item()
            mask = n == names
            if not mask.any():
                continue
            this_cls_size = e['annos']['dimensions'][mask]
            cls_dict[n].append(this_cls_size)

    for name, sizes in cls_dict.items():
        cls_dict[name] = np.concatenate(sizes, 0)

    stat_list = []
    for name, sizes in cls_dict.items():
        mean_size = sizes.mean(0)
        num = len(sizes)
        # print(f'Number of {name:<{35}}: {num:<{10}}average size: {mean_size}')
        stat_list.append((name, num, mean_size))

    stat_list = sorted(stat_list, key=lambda x: -x[1])
    for s in stat_list:
        print(f'Number of {s[0]:<{35}}: {s[1]:<{10}}average size: {s[2]}')



if __name__ == '__main__':
    info_path = './data/argo2/kitti_format/argo2_infos_val.pkl'

    with open(info_path, 'rb') as fr:
        infos = pkl.load(fr)

    get_average_sizes(infos)
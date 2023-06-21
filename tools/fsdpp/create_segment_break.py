import pickle as pkl
from ipdb import set_trace

old_segment_break=[0,4931,9879,14820,19764,24719,29680,34833,39987]

if __name__ == '__main__':
    split = 'test'
    info_path = f'./data/waymo/kitti_format/waymo_infos_{split}.pkl'
    with open(info_path, 'rb') as fr:
        infos = pkl.load(fr)

    breaks = []
    for i, info in enumerate(infos):
        idx = info['image']['image_idx']
        frame_id = f'{idx:07d}'[-3:]
        if frame_id == '000':
            breaks.append(i)

    seg_intervals = {'train':100, 'test':19}
    breaks_per_gpu = [frame_id for seg_id, frame_id in enumerate(breaks) if seg_id % seg_intervals[split] == 0]

    breaks_per_gpu.append(len(infos))
    print(breaks_per_gpu)

    assert len(breaks_per_gpu) == 9
    for i, frame_id in enumerate(breaks_per_gpu):
        if i < 8:
            sample_idx = infos[frame_id]['image']['image_idx']
            assert sample_idx % 1000 == 0
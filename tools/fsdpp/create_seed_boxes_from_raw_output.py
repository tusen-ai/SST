import os
import numpy as np
import pickle as pkl
import tqdm
from ipdb import set_trace

if __name__ == '__main__':
    config = 'fsd_inc_5f_good_vel_copypaste'
    split = 'val'
    raw_path = f'./work_dirs/{config}/raw_{split}_results.pkl'
    save_path = f'./data/waymo/kitti_format/{config}_seed_prediction_{split}.pkl'
    assert not os.path.exists(save_path)
    out_dict = {}

    with open(raw_path, 'rb') as fr:
        raw_output = pkl.load(fr)

    for result in tqdm.tqdm(raw_output):
        boxes = result['boxes_3d']
        scores = result['scores_3d']
        labels = result['labels_3d']
        if len(boxes) == 0:
            continue

        assert labels.max().item() <= 2, 'Holds in Waymo'

        names = ['Car', 'Pedestrian', 'Cyclist']
        label_list = labels.tolist()

        gt_names = [names[l] for l in label_list]
        gt_names = np.array(gt_names, dtype='<U32')


        sample_idx = result['sample_idx']
        idx_str = f'{sample_idx:07d}'

        out_dict[idx_str] = dict(
            gt_bboxes_3d=boxes,
            scores=scores,
            gt_names=gt_names,
        )

    with open(save_path, 'wb') as fw:
        pkl.dump(out_dict, fw)

    print(f'Saved to {save_path}')
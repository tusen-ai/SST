import torch
import os
# from mmdet3d.utils import vis_bev_pc, vis_bev_pc_list
from mmdet3d.datasets import build_dataset
from ipdb import set_trace
import tqdm
import pickle as pkl

if __name__ == '__main__':

    dataset_type = 'WaymoDataset'
    data_root = 'data/waymo/kitti_format/'
    file_client_args = dict(backend='disk')
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    split = 'train'
    assert split in ['train', 'val']

    pipeline = [
        # dict(
        #     type='LoadPointsFromFile',
        #     coord_type='LIDAR',
        #     load_dim=6,
        #     use_dim=5,
        #     file_client_args=file_client_args),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            file_client_args=file_client_args),
    ]

    dataset_cfg = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f'waymo_infos_{split}.pkl',
        split='training',
        pipeline=pipeline,
        modality = dict(use_lidar=True, use_camera=False),
        classes=class_names,
        test_mode=False,
    )

    dataset = build_dataset(dataset_cfg)

    # pc_idx = 10
    save_path = os.path.join(data_root, f'gt_seed_prediction_{split}.pkl')
    seed_prediction_info = {}
    N = len(dataset)
    print(f'Start generating {N} samples ...')
    for i in tqdm.tqdm(range(N)):
        input_dict = dataset.get_data_info(i)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        pts_filename = example['pts_filename']
        sample_idx = pts_filename.split('/')[-1].split('.')[0]
        gt_bboxes_3d = example['gt_bboxes_3d'].tensor.numpy()
        gt_labels_3d = example['gt_labels_3d']
        gt_names = example['ann_info']['gt_names']

        this_sample = {}

        this_sample['idx_in_info'] = i
        this_sample['gt_bboxes_3d'] = gt_bboxes_3d
        this_sample['gt_labels_3d'] = gt_labels_3d
        this_sample['gt_names'] = gt_names

        seed_prediction_info[sample_idx] = this_sample

    with open(save_path, 'wb') as fw:
        pkl.dump(seed_prediction_info, fw)
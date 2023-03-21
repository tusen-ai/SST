class_names = \
['Regular_vehicle',
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
group1 = class_names[:1]
group2 = class_names[1:5]
group3 = class_names[5:11]
group4 = class_names[11:20]
group5 = class_names[20:25]
group6 = class_names[25:]
assert len(group6) == 1
sample_group_1 = {k:1 for k in group1}
sample_group_2 = {k:2 for k in group2}
sample_group_3 = {k:2 for k in group3}
sample_group_4 = {k:1 for k in group4}
sample_group_5 = {k:2 for k in group5}
sample_group_6 = {k:2 for k in group6}
#merge all groups
sample_groups = {**sample_group_1, **sample_group_2, **sample_group_3, **sample_group_4, **sample_group_5, **sample_group_6}
sample_groups.update({'Wheelchair':0, 'Dog':0, 'Message_board_trailer':0})
# dataset settings
# D5 in the config name means the whole dataset is divided into 5 folds
# We only use one fold for efficient experiments
dataset_type = 'Argo2Dataset'
data_root = 'data/argo2/kitti_format/'
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://waymo_data/'))


point_cloud_range = [-204.8, -204.8, -3.2, 204.8, 204.8, 3.2]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'argo2_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points={k:5 for k in class_names}),
    classes=class_names,
    sample_groups=sample_groups,
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            # ann_file=data_root + 'waymo_infos_val_mini.pkl',
            ann_file=data_root + 'argo2_infos_train.pkl',
            split='training',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            # load one frame every five frames
            load_interval=1)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'argo2_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'argo2_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

evaluation = dict(interval=24, pipeline=eval_pipeline)

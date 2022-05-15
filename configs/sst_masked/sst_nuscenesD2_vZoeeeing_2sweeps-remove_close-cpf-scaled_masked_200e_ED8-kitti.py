_base_ = [
    '../sst_refactor/sst_2sweeps_VS0.5_WS16_ED8_epochs288.py'
]
use_chamfer, use_num_points, use_fake_voxels = True, True, True
relative_error = False
masking_ratio = 0.7
fake_voxels_ratio = 0.1
loss_weights = dict(
    loss_occupied=1.0,
    loss_num_points_masked=1.,
    loss_chamfer_src_masked=1.,
    loss_chamfer_dst_masked=1.,
    loss_num_points_unmasked=0.,
    loss_chamfer_src_unmasked=0.,
    loss_chamfer_dst_unmasked=0.
)
window_shape = (16, 16, 1) # 12 * 0.32m
drop_info_training = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info_test = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info = (drop_info_training, drop_info_test)

model = dict(
    type='DynamicVoxelNet',

    voxel_encoder=dict(
        return_gt_points=True
    ),

    middle_encoder=dict(
        _delete_=True,
        type='SSTInputLayerV2Masked',
        window_shape=window_shape,
        sparse_shape=(400, 400, 1),
        voxel_size=(0.25, 0.25, 8),
        shuffle_voxels=True,
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        mute=True,
        masking_ratio=masking_ratio,
        drop_points_th=100,
        pred_dims=3,  # x, y, z
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
        fake_voxels_ratio=fake_voxels_ratio
    ),

    backbone=dict(
        type='SSTv2',
        num_attached_conv=0,
        masked=True
    ),

    neck=dict(
        _delete_=True,
        type='SSTv2Decoder',
        d_model=[128, ] * 6,
        nhead=[8, ] * 6,
        num_blocks=6,
        dim_feedforward=[256, ] * 6,
        output_shape=[400, 400],
        debug=True,
        use_fake_voxels=use_fake_voxels,
    ),

    bbox_head=dict(
        _delete_=True,
        type='ReconstructionHead',
        in_channels=128,
        feat_channels=128,
        num_chamfer_points=10,
        pred_dims=3,
        only_masked=True,
        relative_error=relative_error,
        loss_weights=loss_weights,
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
    )
)

# Dataset

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-49.95, -49.95, -4.95, 49.95, 49.95, 2.95]
number_of_sweeps = 1  # Extra sweeps to be merged. Max is 10.

## NuScenes
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
dataset_type = 'NuScenesDataset'
data_root = './data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))
train_pipeline_nuscences = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=2.0,
        test_mode=True,),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=2.0,
        test_mode=True,),
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
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=2.0,
        test_mode=True,),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
train_nusc = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline_nuscences,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        load_interval=2,  # 1/2
    ),
)
val = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    test_mode=True,
    box_type_3d='LiDAR'
)
test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    test_mode=True,
    box_type_3d='LiDAR'
)

## KITTI
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
# point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6))

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://kitti_data/'))

train_pipeline_kitti = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args,
        kitti=True
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
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

train_kitti = dict(
    type='RepeatDataset',
    times=4,  # Contains 3712 training samples compared to nuscenes 28144 which is 7.58 times larger
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_train.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=train_pipeline_kitti,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'
    )
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='ConcatDataset',
        datasets=[train_nusc, train_kitti],
        separate_eval=False
    ),
    val=val,
    test=test
)

# This schedule is mainly used by models with dynamic voxelization
# optimizer
lr = 0.0005  # max learning rate
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),  # the momentum is change during training
    weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-7)

momentum_config = None

# runtime settings
epochs = 200
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
evaluation = dict(interval=epochs+1)  # Don't evaluate when doing pretraining
workflow = [("train", 1), ("val", 1)]  # But calculate val loss after each epoch
checkpoint_config = dict(interval=20)

fp16 = dict(loss_scale=32.0)
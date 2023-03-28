_base_ = [
    '../_base_/datasets/argo2-3d-26class.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

seg_voxel_size = (0.2, 0.2, 0.2)
point_cloud_range = [-204.8, -204.8, -3.2, 204.8, 204.8, 3.2]
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
num_classes = len(class_names)

seg_score_thresh = [0.4, 0.25, 0.25, 0.25, 0.25, 0.25]
group_lens = [len(group1), len(group2), len(group3), len(group4), len(group5), len(group6)]

model = dict(
    type='VoteSegmentor',

    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),

    middle_encoder=dict(
        type='PseudoMiddleEncoderForSpconvFSD',
    ),

    backbone=dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=[32, 2048, 2048],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        decoder_channels=((128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 0), (1, 0), (0, 0), (0, 1)),
    ),

    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),

    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=131 - 64,
        hidden_dims=[128, 128],
        num_classes=26,
        dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='naiveSyncBN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[1.0, ] * 26 + [0.1,], 
            loss_weight=3.0),
        loss_vote=dict(
            type='L1Loss',
            loss_weight=1.0),
    ),
    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh, # no 
        class_names=class_names, 
        group_names=[group1, group2, group3, group4, group5, group6],
        group_lens=group_lens,
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=6)
evaluation = dict(interval=6)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            load_interval=5)
    ),
)
log_config=dict(
    interval=50,
)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

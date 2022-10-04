# train the segmentor with a short schedule (20% training data)
_base_ = [
    '../_base_/datasets/waymo-fsd.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

seg_voxel_size = (0.32, 0.32, 6)
seg_window_shape=(12, 12, 1) # 12 * 0.32m
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
seg_drop_info_training ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100000)},
}
seg_drop_info_test ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100)},
    3:{'max_tokens':144, 'drop_range':(100, 100000)},
}
seg_drop_info = (seg_drop_info_training, seg_drop_info_test)
class_names = ['Car', 'Pedestrian', 'Cyclist']
num_classes = len(class_names)
seg_score_thresh = (0.5, 0.25, 0.25)

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
        in_channels=5,
        feat_channels=[64, 64, 128],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        unique_once=True,
    ),

    middle_encoder=dict(
        type='SSTInputLayerV2',
        window_shape=seg_window_shape,
        sparse_shape=(468, 468, 1),
        shuffle_voxels=True,
        debug=True,
        drop_info=seg_drop_info,
        pos_temperature=1000,
        normalize_pos=False,
    ),

    backbone=dict(
        type='SSTv2',
        d_model=[128,] * 4,
        nhead=[8, ] * 4,
        num_blocks=4,
        dim_feedforward=[256, ] * 4,
        num_attached_conv=0,
        conv_in_channel=128,
        conv_out_channel=128,
        debug=True,
        to_bev=False,
        layer_cfg=dict(use_bn=True, cosine=True, tau_min=0.01),
    ),

    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),

    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=67 + 64,
        hidden_dims=[128, 128],
        num_classes=num_classes,
        dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='naiveSyncBN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.8,
            loss_weight=1.0),
        loss_vote=dict(
            type='L1Loss',
            loss_weight=1.0),
    ),
    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh, # for training log
        class_names=('Car', 'Ped', 'Cyc'), # for training log
        centroid_offset=False,
    ),
)


# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12)

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

optimizer = dict(
    lr=2e-5,
)
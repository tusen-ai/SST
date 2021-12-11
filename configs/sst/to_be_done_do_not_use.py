# config of sparse-convolution-like API, not finish yet.
_base_ = [
    '../_base_/datasets/waymoD20-3d-3class.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

# model settings
# voxel_size_list = [(0.64, 0.64, 6), (0.32, 0.32, 6), (0.16, 0.16, 6)]
voxel_size_list = [(0.32, 0.32, 6), ]
window_shape=(12, 12, 1) # window_size = 5.12m x 5.12m, 32x32 grid
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
drop_info_training ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100000)},
}
drop_info_test ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100)},
    3:{'max_tokens':144, 'drop_range':(100, 100000)},
}
drop_info = (drop_info_training, drop_info_test)
shifts_list=[(0, 0), (window_shape[0]//2, window_shape[1]//2) ]

model = dict(
    type='DynamicVoxelNet',

    voxel_layer=dict(
        voxel_size=voxel_size_list[0],
        max_num_points=20, # voxel contains points less than 20 will be leaf node
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=3,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size_list[0],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),

    middle_encoder=dict(
        type='IdentityMiddleEncoder',
    ),

    backbone=dict(
        type='SST',
        d_model=[128,] * 4,
        nhead=[8, ] * 4,
        num_blocks=4,
        dim_feedforward=[256, ] * 4,
        dropout=0.0,
        activation="gelu",
        output_shape=[468, 468],
        num_attached_conv=3,
        conv_kwargs=[
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=2, padding=1, stride=1),
        ],
        conv_in_channel=128,
        conv_out_channel=128,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        debug=True,
        batching_info=drop_info,
        no_pos_embed=False,
        pos_temperature=10000,
        normalize_pos=False,
        window_shape=window_shape,
        init_sparse_shape=(468, 468, 1),
        # checkpoint_blocks=[0,],
    ),

    neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[128,],
        upsample_strides=[1,],
        out_channels=[384, ]
    ),

    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345],
                    [-74.88, -74.88, -0.1188, 74.88, 74.88, -0.1188],
                    [-74.88, -74.88, 0, 74.88, 74.88, 0]],
            sizes=[
                [2.08, 4.73, 1.77],  # car
                [0.84, 1.81, 1.77],  # cyclist
                [0.84, 0.91, 1.74]  # pedestrian
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.5),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            dict(  # cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            dict(  # pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=4096,
            nms_thr=0.25,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=500
    )
)

lr=1e-5
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),  # the momentum is change during training
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    )
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(100, 1e-3),
    cyclic_times=1,
    step_ratio_up=0.1,
)
momentum_config = None

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)

fp16 = dict(loss_scale=32.0)
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
evaluation = dict(interval=12)
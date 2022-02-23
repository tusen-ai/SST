_base_ = [
    '../_base_/models/sst_base.py',
    '../_base_/datasets/waymo-3d-car.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

voxel_size = (0.32, 0.32, 6)
window_shape=(12, 12) # 12 * 0.32m
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
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=3,
        feat_channels=[64, 64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),

    middle_encoder=dict(
        type='SSTInputLayer',
        window_shape=window_shape,
        shifts_list=shifts_list,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        shuffle_voxels=True,
        debug=True,
        drop_info=drop_info,
    ),

    backbone=dict(
        type='SSTv1',
        d_model=[128,] * 6,
        nhead=[8, ] * 6,
        num_blocks=6,
        dim_feedforward=[256, ] * 6,
        output_shape=[468, 468],
        num_attached_conv=3,
        conv_kwargs=[
            dict(kernel_size=3, dilation=2, padding=2, stride=1),
            dict(kernel_size=3, dilation=2, padding=2, stride=1),
            dict(kernel_size=3, dilation=2, padding=2, stride=1),
        ],
        conv_in_channel=128,
        conv_out_channel=128,
        debug=True,
        drop_info=drop_info,
        pos_temperature=1000,
        normalize_pos=False,
        window_shape=window_shape,
        checkpoint_blocks=[0,1]
    ),
    bbox_head=dict(
        num_classes=3,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345],
            ],
            sizes=[
                [2.08, 4.73, 1.77],  # car
            ],
            rotations=[0, 1.57],
            reshape_out=False
        ),
    ),
    train_cfg=dict(
        assigner=[
            dict(  # car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
        ],
    ),
    test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=4096,
            nms_thr=0.25,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=500,
            wnms_thr_hi=0.7,
            wnms_thr_lo=0.1,
            wnms=True
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12)

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
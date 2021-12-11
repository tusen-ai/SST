_base_ = [
    '../_base_/models/sst_base.py',
    '../_base_/datasets/waymo-3d-ped-cyc-3sweep.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

voxel_size = (0.32, 0.32, 6)
window_shape=(10, 10) # 12 * 0.32m
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
drop_info_training ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100000)},
}
drop_info_test ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100000)},
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
        in_channels=4,
        feat_channels=[64, 128],
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
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
        ],
        conv_in_channel=128,
        conv_out_channel=128,
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        window_shape=window_shape,
        checkpoint_blocks=[0,1,2]
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=2,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-74.88, -74.88, -0.1188, 74.88, 74.88, -0.1188],
                [-74.88, -74.88, 0, 74.88, 74.88, 0]
            ],
            sizes=[
                [0.84, 1.81, 1.77],  # cyclist
                [0.84, 0.91, 1.74]  # pedestrian
            ],
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
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
    ),
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
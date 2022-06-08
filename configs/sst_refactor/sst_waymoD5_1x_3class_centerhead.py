# one more conv, 5 blocks
_base_ = [
    '../_base_/models/sst_base.py',
    '../_base_/datasets/waymo-3d-3class-5dim.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

voxel_size = (0.32, 0.32, 6)
window_shape=(12, 12, 1) # 12 * 0.32m
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

model = dict(
    type='DynamicCenterPoint',

    voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=5,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),

    middle_encoder=dict(
        type='SSTInputLayerV2',
        window_shape=window_shape,
        sparse_shape=(468, 468, 1),
        shuffle_voxels=True,
        debug=True,
        drop_info=drop_info,
        pos_temperature=1000,
        normalize_pos=False,
    ),

    backbone=dict(
        type='SSTv2',
        d_model=[128,] * 4,
        nhead=[8, ] * 4,
        num_blocks=4,
        dim_feedforward=[256, ] * 4,
        output_shape=[468, 468],
        num_attached_conv=4,
        conv_kwargs=[
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=2, padding=2, stride=1),
        ],
        conv_in_channel=128,
        conv_out_channel=128,
        debug=True,
        layer_cfg=dict(use_bn=False, cosine=True, tau_min=0.01),
        checkpoint_blocks=[0, 1], # Consider removing it if the GPU memory is suffcient
        conv_shortcut=True,
    ),
    neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[128,],
        upsample_strides=[1,],
        out_channels=[128, ]
    ),


    bbox_head=dict(
        type='CenterHead',
        _delete_=True,
        in_channels=128,
        tasks=[
            dict(num_class=3, class_names=['car', 'pedestrian', 'cyclist']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-74.88, -74.88, -10.0, 74.88, 74.88, 10.0],
            max_num=4096,
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=9),
        separate_head=dict(
            type='DCNSeparateHead', init_bias=-2.19, final_kernel=3,
            dcn_config=dict(
                type='DCN',
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4,
                bias=False
            ),  # mmcv 1.2.6 doesn't support bias=True anymore
            norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        ),
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2),
        norm_bbox=True
    ),
    # model training and testing settings
    train_cfg=dict(
        grid_size=[468, 468, 1],
        voxel_size=voxel_size,
        out_size_factor=1,
        dense_reg=1, # not used
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        point_cloud_range=point_cloud_range,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    ),
    test_cfg=dict(
        post_center_limit_range=[-80, -80, -10, 80, 80, 10],
        max_per_img=500, # what is this
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175], # not used in normal nms, task-wise
        score_threshold=0.1,
        pc_range=point_cloud_range[:2], # seems not used
        out_size_factor=1,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        pre_max_size=4096,
        post_max_size=500,
        nms_thr=0.7
    )
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
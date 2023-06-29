_base_ = [
    # '../_base_/datasets/argo2-3d-26class-debug.py',
    '../_base_/datasets/argo2-3d-26class.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

seg_voxel_size = (0.2, 0.2, 0.2)
point_cloud_range = [-204.8, -204.8, -3.2, 204.8, 204.8, 3.2]
virtual_voxel_size=(0.4, 0.4, 0.4) #(1024, 1024, 16)

sparse_shape=[32, 2048, 2048]

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

segmentor = dict(
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
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128, # dummy
        encoder_channels=((128, ), (128, 128, ), (128, 128, ), (128, 128, 128), (256, 256, 256), (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, ), (1, 1, ), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        decoder_channels=((256, 256, 256), (256, 256, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1), (1, 1)), # decoder paddings seem useless in SubMConv
        return_multiscale_features=True,
    ),

    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),

    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=131,
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

model = dict(
    type='SingleStageFSDV2',

    segmentor=segmentor,

    virtual_point_projector=dict(
        in_channels=98 + 64,
        hidden_dims=[64, 64],
        norm_cfg=dict(type='naiveSyncBN1d'),

        ori_in_channels=67 + 64,
        ori_hidden_dims=[64, 64],

        # recover_in_channels=128 + 3, # with point2voxel offset
        # recover_hidden_dims=[128, 128],
    ),

    multiscale_cfg=dict(
        multiscale_levels=[0, 1, 2],
        projector_hiddens=[[256, 128], [128, 128], [128, 128]],
        fusion_mode='avg',
        target_sparse_shape=[16, 1024, 1024],
        norm_cfg=dict(type='naiveSyncBN1d'),
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=67,
        feat_channels=[64, 128],
        voxel_size=virtual_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        unique_once=True,
    ),

    backbone=dict(
        type='VirtualVoxelMixer',
        in_channels=128,
        sparse_shape=[16, 1024, 1024],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, ), (64, 64, ), ),
        encoder_paddings=((1, ), (1, 1,), (1, 1,), ),
        decoder_channels=((64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 1), (1, 1),), # decoder paddings seem useless in SubMConv
    ),

    bbox_head=dict(
        type='FSDV2Head',
        num_classes=num_classes,
        bbox_coder=dict(type='BasePointBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.0,
            alpha=0.25,
            loss_weight=4.0),
        loss_center=dict(type='SmoothL1Loss', loss_weight=0.25, beta=0.1),
        loss_size=dict(type='SmoothL1Loss', loss_weight=0.25, beta=0.1),
        loss_rot=dict(type='SmoothL1Loss', loss_weight=0.1, beta=0.1),
        in_channel=128,
        shared_mlp_dims=[256, 256],
        train_cfg=None,
        test_cfg=None,
        norm_cfg=dict(type='LN'),
        tasks=[
            dict(class_names=group1),
            dict(class_names=group2),
            dict(class_names=group3),
            dict(class_names=group4),
            dict(class_names=group5),
            dict(class_names=group6),
        ],
        class_names=class_names,
        common_attrs=dict(
            center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128),  # (out_dim, num_layers, hidden_dim)
        ),
        num_cls_layer=2,
        cls_hidden_dim=128,
        separate_head=dict(
            type='FSDSeparateHead',
            norm_cfg=dict(type='LN'),
            act='relu',
        ),

    ),

    train_cfg=dict(
        score_thresh=seg_score_thresh,
        sync_reg_avg_factor=True,
        batched_group_sample=True,
        offset_weight='max',
        class_names=class_names,
        group_names=[group1, group2, group3, group4, group5, group6],
        centroid_assign=True,
        disable_pretrain=True,
        disable_pretrain_topks=[300, ] * num_classes,
    ),
    test_cfg=dict(
        score_thresh=seg_score_thresh,
        batched_group_sample=True,
        offset_weight='max',
        class_names=class_names,
        group_names=[group1, group2, group3, group4, group5, group6],
        use_rotate_nms=True,
        nms_pre=-1,
        nms_thr=0.15,
        score_thr=0.05, 
        min_bbox_size=0,
        max_num=500,
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=24)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            load_interval=1)
    ),
)
log_config=dict(
    interval=50,
)
# load_from='./data/pretrain/argo_segmentation_pretrain.pth'
lr=2e-5
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),  # the momentum is change during training
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
)
custom_hooks = [
    dict(type='DisableAugmentationHook', num_last_epochs=4, skip_type_keys=('ObjectSample',)),
    dict(type='EnableFSDDetectionHookIter', enable_after_iter=3000, threshold_buffer=0.3, buffer_iter=6000) 
]
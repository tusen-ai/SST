_base_ = [
    '../_base_/datasets/nusc-10class.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py',
]

seg_voxel_size = (0.2, 0.2, 0.2)
virtual_voxel_size=(0.4, 0.4, 0.4) #(1024, 1024, 16)
point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
sparse_shape = [40, 512, 512]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

num_classes = len(class_names)
group1 = ['car']
group2 = ['truck', 'construction_vehicle']
group3 = ['bus', 'trailer']
group4 = ['barrier']
group5 = ['motorcycle', 'bicycle']
group6 = ['pedestrian', 'traffic_cone']
group_names=[group1, group2, group3, group4, group5, group6]

seg_score_thresh = [0.2, ] * 3 + [0.1, ] * 3
group_lens = [len(group1), len(group2), len(group3), len(group4), len(group5), len(group6)]

head_group1 = class_names[:5]
head_group2 = class_names[5:]
tasks=[
    dict(class_names=head_group1),
    dict(class_names=head_group2),
]

segmentor = dict(
    type='VoteSegmentor',
    tanh_dims=[],
    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 64],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        unique_once=True,
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
        in_channel=67 + 64,
        hidden_dims=[128, 128],
        num_classes=num_classes,
        dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='naiveSyncBN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[1.0, ] * num_classes + [0.1,], 
            loss_weight=10.0),
        loss_vote=dict(
            type='L1Loss',
            loss_weight=1.0),
    ),
    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh, # for training log
        class_names=class_names, # for training log
        group_names=group_names,
        group_lens=group_lens,
    ),
)

model = dict(
    type='SingleStageFSDV2',

    segmentor=segmentor,

    virtual_point_projector=dict(
        in_channels=83 + 64,
        hidden_dims=[64, 64],
        norm_cfg=dict(type='naiveSyncBN1d'),

        ori_in_channels=67 + 64,
        ori_hidden_dims=[64, 64],
    ),

    multiscale_cfg=dict(
        multiscale_levels=[0, 1, 2],
        projector_hiddens=[[256, 128], [128, 128], [128, 128]],
        fusion_mode='avg',
        target_sparse_shape=[20, 256, 256],
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
        sparse_shape=[20, 256, 256],
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
        bbox_coder=dict(type='BasePointBBoxCoder', code_size=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=4.0),
        loss_center=dict(type='L1Loss', loss_weight=0.5),
        loss_size=dict(type='L1Loss', loss_weight=0.5),
        loss_rot=dict(type='L1Loss', loss_weight=0.2),
        loss_vel=dict(type='L1Loss', loss_weight=0.2),
        in_channel=128,
        shared_mlp_dims=[256, 256],
        train_cfg=None,
        test_cfg=None,
        norm_cfg=dict(type='naiveSyncBN1d'),
        tasks=tasks,
        class_names=class_names,
        common_attrs=dict(
            center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128), vel=(2, 2, 128)  # (out_dim, num_layers, hidden_dim)
        ),
        num_cls_layer=2,
        cls_hidden_dim=128,
        separate_head=dict(
            type='FSDSeparateHead',
            norm_cfg=dict(type='naiveSyncBN1d'),
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
        disable_pretrain_topks=[500, ] * num_classes,
    ),
    test_cfg=dict(
        score_thresh=seg_score_thresh,
        batched_group_sample=True,
        offset_weight='max',
        class_names=class_names,
        group_names=[group1, group2, group3, group4, group5, group6],
        use_rotate_nms=True,
        nms_pre=-1,
        nms_thr=0.25,
        # score_thr=0.1, 
        score_thr=0.05, 
        min_bbox_size=0,
        max_num=500,
        all_task_max_num=500,
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12)

log_config=dict(
    interval=50,
)
custom_hooks = [
    dict(type='DisableAugmentationHook', num_last_epochs=3, skip_type_keys=('ObjectSample',), dataset_wrap=True),
    dict(type='EnableFSDDetectionHookIter', enable_after_iter=4000, threshold_buffer=0.4, buffer_iter=8000) 
]

data = dict(
    samples_per_gpu=2, #bz==2
)

optimizer = dict(
    lr=2e-4,
)
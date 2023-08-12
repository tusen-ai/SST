# old name: trk_ped_width1_2x
_base_ = [
    '../_base_/datasets/waymo-tracklet-vehicle.py', # use vehicle base config, it does not matter since overwrite the data pipeline
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

seg_voxel_size = (0.2, 0.2, 0.2)
point_cloud_range = [-204.8, -204.8, -4.0, 204.8, 204.8, 8.0]
class_names = ['Pedestrian',]
num_classes = len(class_names)

segmentor = dict(
    type='TrackletSegmentor',
    tanh_dims=[3, 4],

    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    timestamp_encoder=dict(
        strategy='scalar',
        normalizer=100,
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=11,
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
        sparse_shape=[64, 2048, 2048], 
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128), (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1), (1, 1, 1)),
        decoder_channels=((256, 256, 128), (128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1)), # decoder paddings seem useless in SubMConv
    ),

    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),

    segmentation_head=None,

)

model = dict(
    type='TrackletDetector',

    segmentor=segmentor,

    roi_head=dict(
        type='TrackletRoIHead',
        num_classes=num_classes,
        general_cfg=dict(
            with_roi_scores=True,
        ),
        roi_extractor=dict(
             type='TrackletPointRoIExtractor',
             extra_wlh=[0.5, 0.5, 0.5],
             max_inbox_point=512,
             max_all_point=(300000, 600000),
             debug=False, 
             combined=True,
        ),
        bbox_head=dict(
            type='FullySparseBboxHead',
            num_classes=num_classes,
            num_blocks=6,
            in_channels=[85, 144, 144, 144, 144, 144], 
            feat_channels=[[128, 128], ] * 6,
            rel_mlp_hidden_dims=[[16, 32],] * 6,
            rel_mlp_in_channels=[13, ] * 6,
            reg_mlp=[512, 512],
            cls_mlp=[512, 512],
            mode='max',
            xyz_normalizer=[20, 20, 4],
            act='gelu',
            geo_input=True,
            with_corner_loss=False,
            corner_loss_weight=1.0,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            norm_cfg=dict(type='LN', eps=1e-3),
            unique_once=True,

            loss_bbox=dict(
                type='L1Loss',
                reduction='mean',
                loss_weight=2.0),

            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0),
            cls_dropout=0.1,
            reg_dropout=0.1,
        ),
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None
    ),

    train_cfg=dict(
        # pre_voxelization_size=(0.1, 0.1, 0.1),
        pre_voxelization_size=None,
        assigner=dict( # Car
            type='TrackletAssigner',
        ),
        hack_sampler_bug=True,
        cls_pos_thr=(0.65, ),
        cls_neg_thr=(0.15, ),

        sync_reg_avg_factor=True,
        sync_cls_avg_factor=True,
        corner_loss_only_car=True, # default True, explicitly set to False to disable
        class_names=class_names,
        rcnn_code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ),
    test_cfg=dict(
        batch_inference=True,
        identical_decode=False,
        # tta=dict(
        #     merge='weighted',
        # ),
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=24)

train_pipeline = [
    dict(
        type='LoadTrackletPoints',
        load_dim=6,
        use_dim=5, # remove the wrong timestamp
        max_points=1024,
        debug=False,
    ),

    dict(
        type='LoadTrackletAnnotations',
    ),

    dict(
        type='TrackletPoseTransform',
        concat=False,
    ),
    dict(
        type='PointDecoration',
        properties=['yaw', 'size', 'score'],
        concat=True,
    ),

    dict(
        type='TrackletRandomFlip',
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5
    ),

    dict(
        type='TrackletGlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0.2]
    ),

    dict(type='PointsRangeFilter', point_cloud_range=[-204.7, -204.7, -3.99, 204.7, 204.7, 7.99]),
    dict(type='PointShuffle'),
    dict(type='TrackletFormatBundle', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_frame_inds', 'tracklet', 'gt_tracklet_candidates'])
]
test_pipeline = [
    dict(
        type='LoadTrackletPoints',
        load_dim=6,
        use_dim=5, # remove the wrong timestamp
        max_points=1024,
        debug=False,
    ),
    dict(
        type='TrackletPoseTransform',
        concat=False,
    ),
    dict(
        type='PointDecoration',
        properties=['yaw', 'size', 'score'],
        concat=True,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=[-204.7, -204.7, -3.99, 204.7, 204.7, 7.99]),
    dict(type='PointShuffle'),
    dict(type='TrackletFormatBundle', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_frame_inds', 'tracklet']),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = test_pipeline

tta_pipeline = [
    dict(
        type='LoadTrackletPoints',
        load_dim=6,
        use_dim=5, # remove the wrong timestamp
        max_points=1024,
        debug=False,
    ),
    dict(
        type='TrackletPoseTransform',
        concat=False,
    ),
    dict(
        type='PointDecoration',
        properties=['yaw', 'size', 'score'],
        concat=True,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        # pts_rots=[0, ],
        # pts_rots=[-4*pi/5, -2*pi/5, 0, 2*pi/5, 4*pi/5, ],
        # pts_rots=[0, 2*pi/5, 4*pi/5, -4*pi/5, -2*pi/5, ], # Note the order if use iou clamp
        # pts_rots=[pi/2, 0, -pi/2],
        flip=True,
        pcd_horizontal_flip=True, # double flip
        pcd_vertical_flip=True,
        transforms=[
            dict(
                type='TrackletGlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]
            ),
            dict(
                type='TrackletRandomFlip',
                flip_ratio_bev_horizontal=0.0,
                flip_ratio_bev_vertical=0.0,
            ),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointShuffle'),
            dict(type='TrackletFormatBundle', class_names=class_names),
            dict(type='Collect3D', keys=['points', 'pts_frame_inds', 'tracklet']),
        ]
    )
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            ann_file='./data/waymo/tracklet_data/fsd_base_ped_training_gt_candidates.pkl', # old name: fsd_pastfuture_ped_width1_training_gt_candidates
            tracklet_proposals_file='./data/waymo/tracklet_data/fsd_base_ped_training.pkl', # old name: fsd_pastfuture_ped_width1_training
            pipeline=train_pipeline,
            classes=class_names,
            load_interval=1,
        )
    ),
    val=dict(
        pipeline=eval_pipeline,
        min_tracklet_points=1,
        samples_per_gpu=8,
        classes=class_names,
        ),
    test=dict(
        tracklet_proposals_file='./data/waymo/tracklet_data/fsd_base_ped_val.pkl', # old name: fsd_pastfuture_ped_width1_val
        pipeline=test_pipeline,
        # pipeline=tta_pipeline,
        min_tracklet_points=1,
        samples_per_gpu=8,
        classes=class_names,
    )
)
log_config=dict(
    interval=50,
)

optimizer = dict(
    lr=1e-5,
)
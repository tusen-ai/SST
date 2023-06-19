_base_ = [
    '../_base_/datasets/waymo-3d-3class-incremental-8f.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

seg_voxel_size = (0.2, 0.2, 0.2)
point_cloud_range = [-80, -80, -2, 80, 80, 4]
class_names = ['Car', 'Pedestrian', 'Cyclist']
num_classes = len(class_names)
seg_score_thresh = (0.5, 0.25, 0.15)

segmentor = dict(
    type='VoteSegmentor',
    segmentation_test=False,
    drop_output=True,
    tanh_dims=[3, 4],
    voxel_downsampling_size=(0.05, 0.05, 0.05),

    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=6,
        feat_channels=[64, 64],
        with_distance=False,
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
        sparse_shape=[32, 800, 800],
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

    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=67,
        hidden_dims=[128, 128],
        num_classes=num_classes, # using focal loss, binary classification
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
    test_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh, # for test
        clustering_voxel_size=(0.5, 0.5, 6), # for test
    )
)

model = dict(
    type='TwoStageIncrementalDetector',

    incremental_cfg=dict(
        voxel_size=(0.25, 0.25, 0.4),
        point_cloud_range=point_cloud_range,
        center_noise=0.0,
        dim_noise=0.0,
        yaw_noise=0.0,
        extra_width=1.0,
        num_previous_frames=6,
        crop_frames=6,
        max_crop_points=128,
        crop_shuffle=True,
        max_age=1,
        num_base_frame=5,
    ),

    segmentor=segmentor,

    backbone=dict(
        type='StackedVFE',
        num_blocks=3,
        in_channels=[85,] + [134, ] * 2,
        feat_channels=[[128, 128], ] * 3,
        rel_mlp_hidden_dims=[[16, 32],] * 3,
        with_rel_mlp=True,
        with_distance=False,
        with_cluster_center=False,
        norm_cfg=dict(type='LN', eps=1e-3),
        mode='max',
        xyz_normalizer=[20, 20, 4],
        use_middle_cluster_feature=True,
        cat_voxel_feats=True,
        pos_fusion='mul',
        act='gelu',
        unique_once=True,
    ),

    bbox_head=dict(
        type='SparseClusterHeadV2',
        num_classes=num_classes,
        bbox_coder=dict(type='BasePointBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_center=dict(type='L1Loss', loss_weight=0.5),
        loss_size=dict(type='L1Loss', loss_weight=0.5),
        loss_rot=dict(type='L1Loss', loss_weight=0.2),
        in_channel=128 * 3 * 2,
        shared_mlp_dims=[1024, 1024],
        train_cfg=None,
        test_cfg=None,
        norm_cfg=dict(type='LN'),
        tasks=[
            dict(class_names=['Car',]),
            dict(class_names=['Pedestrian',]),
            dict(class_names=['Cyclist',]),
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
        as_rpn=True,
    ),
    roi_head=dict(
        type='GroupCorrectionHead',
        num_classes=num_classes,
        roi_extractor=dict(
             type='DynamicPointROIExtractor',
             extra_wlh=[0.5, 0.5, 0.5],
             max_inbox_point=256,
             debug=False,
        ),
        bbox_head=dict(
            type='FullySparseBboxHead',
            num_classes=num_classes,
            num_blocks=3,
            in_channels=[214, 131+13+3, 131+13+3], 
            feat_channels=[[128, 128], ] * 3,
            with_distance=False,
            with_cluster_center=False,
            with_rel_mlp=True,
            rel_mlp_hidden_dims=[[16, 32],] * 3,
            rel_mlp_in_channels=[13, ] * 3,
            reg_mlp=[512, 512],
            cls_mlp=[512, 512],
            mode='max',
            xyz_normalizer=[20, 20, 4],
            cat_voxel_feats=True,
            pos_fusion='mul',
            fusion='cat',
            act='gelu',
            geo_input=True,
            use_middle_cluster_feature=True,
            with_corner_loss=True,
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
        ),
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        checkpointing=False,
    ),

    train_cfg=dict(
        use_voting_center=True,
        use_gt_assigner=False,
        score_thresh=seg_score_thresh,
        sync_reg_avg_factor=True,
        pre_voxelization_size=(0.1, 0.1, 0.1),
        disable_pretrain=True,
        disable_pretrain_topks=[1200, 400, 400],
        point_drop_ratio=0.05,
        rpn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=None,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=500,
        ),
        rcnn=dict(
            assigner=[
                dict( # Car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1
                ),
                dict( # Cyc
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict( # Ped
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
            ],

            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=256 * 2,
                pos_fraction=0.55,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True
            ),
            cls_pos_thr=(0.8, 0.65, 0.65),
            cls_neg_thr=(0.2, 0.15, 0.15),
            sync_reg_avg_factor=True,
            sync_cls_avg_factor=True,
            corner_loss_only_car=True, # default True, explicitly set to False to disable
            class_names=class_names,
        )
    ),
    test_cfg=dict(
        use_voting_center=True,
        score_thresh=seg_score_thresh,
        pre_voxelization_size=(0.1, 0.1, 0.1),
        # pre_voxelization_size=None,
        skip_rcnn=False,
        sequential=True,
        reuse_test=True,
        reuse_results=True,
        calib_interval=-1,
        rpn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.25,
            score_thr=0.1, 
            min_bbox_size=0,
            max_num=500,
        ),
        rcnn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.25, # from 0.25 to 0.7 for retest
            score_thr=0.1, 
            min_bbox_size=0,
            max_num=500,
        ),
    ),
    cluster_assigner=dict(
        cluster_voxel_size=dict(
            Car=(0.3, 0.3, 6),
            Cyclist=(0.2, 0.2, 6),
            Pedestrian=(0.05, 0.05, 6),
        ),
        min_points=2,
        point_cloud_range=point_cloud_range,
        connected_dist=dict(
            Car=0.6,
            Cyclist=0.4,
            Pedestrian=0.1,
        ), # xy-plane distance
        class_names=class_names,
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12)

# fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            seed_info_path='./data/waymo/kitti_format/fsd_inc_5f_good_vel_copypaste_seed_prediction_train.pkl',
            load_interval=1)
    ),
    test=dict(
        seed_info_path='./data/waymo/kitti_format/fsd_inc_5f_good_vel_copypaste_seed_prediction_val.pkl',
    )
)
log_config=dict(
    interval=50,
)
custom_hooks = [
    dict(type='DisableAugmentationHook', num_last_epochs=1, skip_type_keys=('ObjectSample', 'RandomFlip3D', 'GlobalRotScaleTrans')),
    dict(type='EnableFSDDetectionHookIter', enable_after_iter=4000, threshold_buffer=0.3, buffer_iter=8000)
]

optimizer = dict(
    lr=3e-5,
)
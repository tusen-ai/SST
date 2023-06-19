dataset_type = 'WaymoTrackletDataset'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')

class_names = ['Car', ]
point_cloud_range = [-204.7, -204.7, -3.99, 204.7, 204.7, 7.99] # TODO

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

    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
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
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='TrackletFormatBundle', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_frame_inds', 'tracklet']),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = test_pipeline

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'fsd6f6e_vehicle_1-5_training_gt_candidates.pkl',
            tracklet_proposals_file=data_root + 'fsd6f6e_vehicle_1-5_training.pkl',
            pose_file=data_root + 'poses.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',
            load_interval=1,
            min_tracklet_points=1,
            )),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=None,
        # tracklet_proposals_file=data_root + 'fsd6f6e_val_tracklets.pkl',
        tracklet_proposals_file=None,
        pose_file=data_root + 'poses.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        min_tracklet_points=1,
        ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=None,
        # tracklet_proposals_file=data_root + 'fsd6f6e_val_tracklets.pkl',
        tracklet_proposals_file=None,
        pose_file=data_root + 'poses.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        min_tracklet_points=1,
        ))

evaluation = dict(interval=24, pipeline=eval_pipeline)
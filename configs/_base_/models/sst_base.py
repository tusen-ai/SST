# Configurations of neck, head and assigner, same as PointPillar
model = dict(
    type='DynamicVoxelNet',

    neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[128,],
        upsample_strides=[1,],
        out_channels=[384, ]
    ),

    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795],
                    [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
                    [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
                    [-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111],
                    [-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072],
                    [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
                    [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965]],
            sizes=[[1.95017717, 4.60718145, 1.72270761],
                   [2.4560939, 6.73778078, 2.73004906],
                   [2.87427237, 12.01320693, 3.81509561],
                   [0.60058911, 1.68452161, 1.27192197],
                   [0.66344886, 0.7256437, 1.75748069],
                   [0.39694519, 0.40359262, 1.06232151],
                   [2.49008838, 0.48578221, 0.98297065]],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0), # Changed from L1Loss with no beta. Changed loss_weight from 0.5
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,   # Changed from initial 4096 
            nms_thr=0.2,    # Changed from initial 0.25
            score_thr=0.05, # Changed from initial 0.1
            min_bbox_size=0,
            max_num=500
    )
)
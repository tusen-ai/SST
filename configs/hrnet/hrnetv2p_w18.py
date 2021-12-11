_base_ = [
    '../_base_/models/pointpillars_472.py',
    '../_base_/datasets/waymoD20-3d-3class.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='DynamicMVXFasterRCNN',

    pts_backbone=dict(
        _delete_=True,
        type='HRNet3D',
        in_channels=64,
        with_cp=True,
        # norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(3, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(3, 3),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(3, 3, 3),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(3, 3, 3, 3),
                num_channels=(18, 36, 72, 144))
        ),
    ),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[18, 36, 72, 144],
        upsample_strides=[1, 2, 4, 8],
        out_channels=[96, 96, 96, 96]
    ),
    # model training and testing settings
    # train_cfg=dict(
    #     _delete_=True,
    #     pts=dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             iou_calculator=dict(type='BboxOverlapsNearest3D'),
    #             pos_iou_thr=0.55,
    #             neg_iou_thr=0.4,
    #             min_pos_iou=0.4,
    #             ignore_iof_thr=-1),
    #         allowed_border=0,
    #         code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #         pos_weight=-1,
    #         debug=False)
    # )
)
# fp16 settings
# fp16 = dict(loss_scale=512.0)
# fp16 = dict(loss_scale=64.0)
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
optimizer = dict(type='AdamW', lr=0.001*0.5, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12)
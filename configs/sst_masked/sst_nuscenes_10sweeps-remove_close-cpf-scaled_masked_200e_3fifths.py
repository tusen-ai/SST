_base_ = [
    '../sst_refactor/sst_10sweeps_VS0.5_WS16_ED8_epochs288_3fiths.py'
]
use_chamfer, use_num_points, use_fake_voxels = True, True, True
relative_error = False
masking_ratio = 0.7
fake_voxels_ratio = 0.1
loss_weights = dict(
    loss_occupied=1.,
    loss_num_points_masked=1.,
    loss_chamfer_src_masked=1.,
    loss_chamfer_dst_masked=1.,
    loss_num_points_unmasked=0.,
    loss_chamfer_src_unmasked=0.,
    loss_chamfer_dst_unmasked=0.
)
window_shape = (16, 16, 1) # 12 * 0.32m
drop_info_training = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info_test = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info = (drop_info_training, drop_info_test)

model = dict(
    type='DynamicVoxelNet',

    voxel_encoder=dict(
        return_gt_points=True
    ),

    middle_encoder=dict(
        _delete_=True,
        type='SSTInputLayerV2Masked',
        window_shape=window_shape,
        sparse_shape=(200, 200, 1),
        voxel_size=(0.5, 0.5, 8),
        shuffle_voxels=True,
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        mute=True,
        masking_ratio=masking_ratio,
        drop_points_th=100,
        pred_dims=3,  # x, y, z
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
        fake_voxels_ratio=fake_voxels_ratio
    ),

    backbone=dict(
        type='SSTv2',
        num_attached_conv=0,
        masked=True
    ),

    neck=dict(
        _delete_=True,
        type='SSTv2Decoder',
        d_model=[128, ] * 6,
        nhead=[8, ] * 6,
        num_blocks=6,
        dim_feedforward=[256, ] * 6,
        output_shape=[200, 200],
        debug=True,
        use_fake_voxels=use_fake_voxels,
    ),

    bbox_head=dict(
        _delete_=True,
        type='ReconstructionHead',
        in_channels=128,
        feat_channels=128,
        num_chamfer_points=10,
        pred_dims=3,
        only_masked=True,
        relative_error=relative_error,
        loss_weights=loss_weights,
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
    )
)


# This schedule is mainly used by models with dynamic voxelization
# optimizer
lr = 0.0005  # max learning rate
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),  # the momentum is change during training
    weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-7)

momentum_config = None

# runtime settings
epochs = 200
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
evaluation = dict(interval=epochs+1)  # Don't evaluate when doing pretraining
workflow = [("train", 1), ("val", 1)]  # But calculate val loss after each epoch
checkpoint_config = dict(interval=epochs//4)

fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

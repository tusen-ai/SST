_base_ = [
    './sst_nuscenesD5_1x_3class_8heads_2v.py.py',
]

window_shape = (16, 16, 1) # 12 * 0.32m
drop_info_training ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100)},
    3:{'max_tokens':200, 'drop_range':(100, 200)},
    4:{'max_tokens':256, 'drop_range':(200, 100000)},
}
drop_info_test ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100)},
    3:{'max_tokens':200, 'drop_range':(100, 200)},
    4:{'max_tokens':256, 'drop_range':(200, 100000)},
}
drop_info = (drop_info_training, drop_info_test)

model = dict(
    type='DynamicVoxelNet',

    voxel_encoder=dict(
        type='DynamicVFE',
        return_gt_points=True
    ),

    middle_encoder=dict(
        _delete_=True,
        type='SSTInputLayerV2Masked',
        window_shape=window_shape,
        sparse_shape=(400, 400, 1),
        shuffle_voxels=True,
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        mute=True,
        masking_ratio=0.7
    ),

    backbone=dict(
        type='SSTv2',
        masked=True
    ),

    neck=dict(
        _delete_=True,
        type='SSTv2Decoder',
        d_model=[128,] * 6,
        nhead=[8, ] * 6,
        num_blocks=6,
        dim_feedforward=[256, ] * 6,
        output_shape=[400, 400],
        debug=True,
    ),

    bbox_head=dict(
        _delete_=True,
        type='ReconstructionHead',
        in_channels=128,
        feat_channels=128,
        num_reg_points=10,
        only_masked=True,
    )
)

# runtime settings
epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
evaluation = dict(interval=epochs+1)  # Don't evaluate when doing pretraining
workflow = [("train", 1), ("val", 1)]  # But calculate val loss after each epoch
checkpoint_config = dict(interval=6)

fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    # Trin with less of the dataset
    #train=dict(
    #    type="NuScenesDataset",
    #    load_interval=5
    #    ),
    # Validate with less of the dataset for speed
    val=dict(
        type="NuScenesDataset",
        load_interval=5
    )
)
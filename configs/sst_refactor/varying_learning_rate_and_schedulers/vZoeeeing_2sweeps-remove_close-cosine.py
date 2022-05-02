# Used to try same settings as Zoeeeing as presented here: https://github.com/TuSimple/SST/issues/18
_base_ = [
    '../_base_/models/sst_base.py',
    '../_base_/datasets/nus-3d-2sweep-remove_close.py',
    '../_base_/schedules/modified/cosine.py',
    '../_base_/default_runtime.py',
]

voxel_size = (0.25, 0.25, 8)
window_shape = (16, 16, 1) # 12 * 0.32m
point_cloud_range = [-50, -50, -5, 50, 50, 3]
drop_info_training ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100)},
    3:{'max_tokens':200, 'drop_range':(100, 200)},
    4:{'max_tokens':250, 'drop_range':(200, 100000)},
}
drop_info_test ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100)},
    3:{'max_tokens':200, 'drop_range':(100, 200)},
    4:{'max_tokens':256, 'drop_range':(200, 100000)},  # 16*16=256
}
drop_info = (drop_info_training, drop_info_test)
shifts_list=[(0, 0), (window_shape[0]//2, window_shape[1]//2)]

model = dict(
    type='DynamicVoxelNet',

    voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
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
        sparse_shape=(400, 400, 1),
        shuffle_voxels=True,
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        mute=True,
    ),

    backbone=dict(
        type='SSTv2',
        d_model=[128,] * 6,
        nhead=[8, ] * 6,
        num_blocks=6,
        dim_feedforward=[256, ] * 6,
        output_shape=[400, 400],
        num_attached_conv=3,
        conv_kwargs=[
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=2, padding=2, stride=1),
        ],
        conv_in_channel=128,
        conv_out_channel=128,
        debug=True,
    ),

)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=24)
checkpoint_config = dict(interval=6)

fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

workflow = [('train', 1), ('val', 1)]  # Includes validation at same frequency as training.

"""train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            load_interval=5)
    ),"""

# Used to try same settings as Zoeeeing as presented here: https://github.com/TuSimple/SST/issues/18
_base_ = [
    './sst_nuscenes_vZoeeeing.py',
    '../_base_/datasets/nus-3d-6sweep-car-remove_close.py',
]

model = dict(
    type='DynamicVoxelNet',

    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
    )
)

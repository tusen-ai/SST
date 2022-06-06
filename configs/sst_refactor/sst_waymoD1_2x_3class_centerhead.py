# one more conv, 5 blocks
_base_ = [
    './sst_waymoD5_1x_3class_centerhead.py',
]

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=24)

data = dict(
    train=dict(
        dataset=dict(load_interval=1)
    ),
)
_base_ = ["./nus-3d-6sweep-remove_close.py"]

train_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=2.0),
]
test_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=2.0
    ),
]
eval_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        pad_empty_sweeps=True,
        remove_close=True,
        close_radius=2.0
    ),
]

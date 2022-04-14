_base_=["./nus-3d-1sweep.py"]

number_of_sweeps = 1  # Extra sweeps to be merged. Max is 6 for now.
train_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,),
]
test_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,),
]
eval_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,
    )
]

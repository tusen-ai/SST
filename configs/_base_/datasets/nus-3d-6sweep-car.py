_base_ = ["./nus-3d-6sweep.py"]
class_names = ['car']
dataset_type = 'NuScenesDataset'
data_root = './data/nuscenes/'
train_pipeline = [
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
]
test_pipeline = [
    dict(
        type='MultiScaleFlipAug3D',
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=class_names,),
    val=dict(
        type=dataset_type,
        classes=class_names,),
    test=dict(
        type=dataset_type,
        classes=class_names,))

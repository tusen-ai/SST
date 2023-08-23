# Instructions of Our Auto-labeling Framework CTRL

### Here we provide a simple guidance to train and inference CTRL. We also offer the processed data for a quick start at the end of this page. ~~Configs of other classes (pedestrian, cyclist) and tricks such as backtracing will be updated in near future.~~ Configs of all classes have been released.


---

## Training

Here we take the vehicle class for example.

### Step 1: Generate train_gt.bin once for all. (waymo bin format).
User could download the generated results 'train_gt.bin' from the link at the end.

`python ./tools/ctrl/generate_train_gt_bin.py`

The generated 'train_gt.bin' will be used in Step 4.

Then extract frame poses and save it:

`python ./tools/ctrl/extract_poses.py`

This step only need to be executed **once** for all experiments on WOD.


### Step 2: Use ImmortalTracker to generate tracking results in training split (bin file format)
~~This step can be finished with any 3D Object Tracking methods, and we will release our modified ImmortalTracker codebase in near future.~~
We have released the tailored ImmortalTracker here: https://github.com/Abyssaledge/ImmortalTracker-for-CTRL

### Step 3: Generate track input for training

This step generates track input for training/inference from the results of base detector (`xxx.bin`). The bin file path and split need to be specified in yaml config file such as the following `fsd_base_vehicle.yaml`.

`python ./tools/ctrl/generate_track_input.py ./tools/ctrl/data_configs/fsd_base_vehicle.yaml --process 8`

### Step 4: Assign candidates GT tracks

This steps select some potential GT track for the input predicted tracks in an offline manner. And the second-round assignment is conducted during training.

`python ./tools/ctrl/generate_candidates.py ./tools/ctrl/data_configs/fsd_base_vehicle.yaml --process 8`

### Step 5: Begin training 

`bash tools/dist_train.sh configs/ctrl/ctrl_veh_24e.py 8 --no-validate`

(--no-validation arguement is required due to a unknow bug)

---

## Inference 

Here we take the vehicle class for example.

### Step 1: Use ImmortalTracker to generate tracking results in bin file format
Similar to step 1 in training

### Step 2 (Optional): Backtracing and Extension, specific information are specified in the extend.yaml
This step is optional. Users could skip this step for quickly start. ~~We will update relevant resources in near future.~~ `extend.yaml` has been updated.

`python ./tools/ctrl/extend_tracks.py ./tools/ctrl/data_configs/extend.yaml`

### Step 3: Generate track input for inference (bin file path and split need to be specified in fsd_base_vehicle.yaml)
`python ./tools/ctrl/generate_track_input.py ./tools/ctrl/data_configs/fsd_base_vehicle.yaml --process 8`

### Step 4: Begin inference (Track TTA is optional, can be enabled in config)
`bash ./tools/dist_test.sh configs/ctrl/ctrl_veh_24e.py ./work_dirs/ctrl_veh_24e/latest.pth 8 --options "pklfile_prefix=./work_dirs/ctrl_veh_24e/result"  --eval waymo`

### Step 5 (Optional): Remove empty predictions
`python ./tools/ctrl/remove_empty.py --bin-path ./$WORK/$CONFIG/result.bin --process 8 --split training --type vehicle`

---

## Resources
We provide pretrained baseline models and result file of base detector in validation and training set, which are required in our pipeline.
Users could use these data for inference, directly from step 3.
Here is the link：https://share.weiyun.com/1r6VsFfJ

or Google Drive link: https://drive.google.com/drive/folders/19-pvKCTLgJ_x6j1C3AvKgHM3GYMNxf6I?usp=sharing

Due to the WOD license, if you are interested in these data for inference, please contact Lue Fan (fanlue2019@ia.ac.cn) and you will be privately authorized for data access. 

## Results
If you use the default config and everything goes well, you should get the following results, which is the baseline result without bidirectional tracking and TTA:

```
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1: [mAP 0.873207] [mAPH 0.866745]
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2: [mAP 0.807134] [mAPH 0.800833]
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_SIGN_LEVEL_1: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_SIGN_LEVEL_2: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2: [mAP 0] [mAPH 0]
RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1: [mAP 0.952167] [mAPH 0.948044]
RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2: [mAP 0.943131] [mAPH 0.939031]
RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1: [mAP 0.871974] [mAPH 0.865268]
RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2: [mAP 0.822372] [mAPH 0.815774]
RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1: [mAP 0.754103] [mAPH 0.741393]
RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2: [mAP 0.635579] [mAPH 0.624029]
```

And the performance of adopted base detector is: 

```
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1: [mAP 0.82786] [mAPH 0.823132]
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2: [mAP 0.751349] [mAPH 0.746877]
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_SIGN_LEVEL_1: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_SIGN_LEVEL_2: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2: [mAP 0] [mAPH 0]
RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1: [mAP 0.93624] [mAPH 0.932365]
RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2: [mAP 0.925796] [mAPH 0.921951]
RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1: [mAP 0.819024] [mAPH 0.813976]
RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2: [mAP 0.761934] [mAPH 0.757116]
RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1: [mAP 0.668922] [mAPH 0.661151]
RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2: [mAP 0.542161] [mAPH 0.535483]
```

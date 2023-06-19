# Instructions of Our Auto-labeling Framework CTRL

### Here we provide a simple guidance to train and inference CTRL. We also offer the processed data for a quick start at the end of this page. Configs of other classes (pedestrian, cyclist) and tricks such as backtracing will be updated in near future.


---

## Training

### Step 1: Generate train_gt.bin once for all. (waymo bin format).
User could download the generated results 'train_gt.bin' from the link at the end.

`python ./tools/ctrl/generate_train_gt_bin.py`

The generated 'train_gt.bin' will be used in Step 4.


### Step 2: Use ImmortalTracker to generate tracking results in training split (bin file format)
This step can be finished with any 3D Object Tracking methods, and we will release our modified ImmortalTracker codebase in near future.

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

### Step 1: Use ImmortalTracker to generate tracking results in bin file format
Similar to step 1 in training

### Step 2 (Optional): Backtracing and Extension, specific information are specified in the extend_veh.yaml
This step is optional. Users could skip this step for quickly start. We will update relevant resources in near future.

`python ./tools/ctrl/extend_tracks.py ./tools/ctrl/data_configs/extend_veh.yaml`

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

Due to the WOD license, if you are interested in these data for inference, please contact Lue Fan (fanlue2019@ia.ac.cn) and you will be offered a private password for data access. 
# Instructions for FSDv2

## Resources
### We provide complete first-hand resources on all three datasets to reproduce our performance and track the training process, including checkpoints, logs, results.
Access the resources at: https://share.weiyun.com/1HgbFpyI or https://drive.google.com/drive/folders/17xG_AVqCOTzPPKl6RNHQyXlwG8hmCyJC?usp=sharing

If the link expires, feel free to open an issue.
Due to the Waymo license, please contact Lue Fan (fanlue2019@ia.ac) to access the Waymo resources privately.

---

### Data Preparation
Follow official MMDetection3Dv0.15.0 to prepare data: https://github.com/open-mmlab/mmdetection3d/releases/tag/v0.15.0 . 
Note that users do not need to install the official MMDetection3D, just following their instructions to prepare data.
If you have used this SST repo before, please skip this step.

### Run Experiments
Users only need to change the content in `run.sh` to:
```
DIR=fsdv2
WORK=work_dirs

# for waymo
CONFIG=fsdv2_waymo_2x
bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/results evaluation.metric=fast --seed 1

# for argoverse 2
CONFIG=fsdv2_argo_2x
bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/results evaluation.metric=fast --seed 1

# for nuscenes
CONFIG=fsdv2_nusc_2x
bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.jsonfile_prefix=./$WORK/$CONFIG/results evaluation.metric=bbox --seed 1
```

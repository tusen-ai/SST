# Instructions of FSD++ (Super Sparse 3D Object Detection)

 **Clarification**: in this repo, we use the iterm `incremental` to name some functions and classes in FSD++, because FSD++ `incrementally` utilizes new observations (residual points).

---

## Inference

### Step 1: Generate seed predictions
Since FSD++ needs initial predictions at the first frame of each sequence, which called `seed predictions`, we should first generate the seed predictions from a WOD results file `xxx.bin`. Given a WOD bin file, use the following command to generate corresponding seed predictions.

`python ./tools/fsdpp/create_seed_boxes_from_bin.py`

Users should specify the data path in the file.

### Step 2: Begin inference

Spicify the path of generated seed predictions in config file (`seed_info_path`). For example, in the given config, we have 
```
data = dict(
    test=dict(
        seed_info_path='./data/waymo/kitti_format/fsd_inc_5f_good_vel_copypaste_seed_prediction_val.pkl',
    )
)
```
And run:

`bash tools/dist_test.sh {your_config} {checkpoint} 8 --options "pklfile_prefix={path_to_save_result}" --eval fast`

If you would like have a try, please contact Lue Fan (fanlue2019@ia.ac.cn), and you will be offered a private passwork to access our generated data and pretrained model in the following link.
https://share.weiyun.com/1fPWqjJK

**Things you'd better to know:**

The inference pipeline of FSD++ is different from normal cases, where we need to reuse the predictions and raw data of previous time stamps. So we must arrange the input frames in time order and all frames of a sequence need being placed in single GPU. We have done all of these in Waymo dataset. However, in other dataset, you have to get things done on your self. If you need help, feel free to open an issue.

If everything goes well, you will get the results on WOD validation set, attached at the end this file.

---

## Training from scratch

### Step 1: Generate seed predictions
Here we adopt the similar strategy to inference that using offline seed predictions as history information.
Since the supported maximum size of WOD bin is 2GB, all predictions in training set may excess the limit. So we first save the raw output of training set by:

`bash tools/dist_test.sh configs/$CONFIG.py ./work_dirs/$CONFIG/latest.pth 8 --options "pklfile_prefix=./work_dirs/$CONFIG/raw_output" --eval raw`

Note the config adopted here is a normal single-frame detector such as FSD, running on the **training set**. Users are supposed to change the config to make the detector run on the training set. Then use the following command to generate seed predictions from the save raw output.

`python ./tools/fsdpp/create_seed_boxes_from_raw_output.py`

Users should change the data path accordingly in `create_seed_boxes_raw_output.py`

### Step 2: Begin training

`bash tools/dist_train.sh {config} 8  --no-validate`

After training, following the inference pipeline to get the results.

---


## Results on WOD validation split

```
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1: [mAP 0.813912] [mAPH 0.809247]
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2: [mAP 0.733229] [mAPH 0.728862]
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1: [mAP 0.851317] [mAPH 0.822419]
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2: [mAP 0.783175] [mAPH 0.754942]
OBJECT_TYPE_TYPE_SIGN_LEVEL_1: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_SIGN_LEVEL_2: [mAP 0] [mAPH 0]
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1: [mAP 0.804813] [mAPH 0.796302]
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2: [mAP 0.782464] [mAPH 0.774163]
RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1: [mAP 0.929272] [mAPH 0.925467]
RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2: [mAP 0.91806] [mAPH 0.914286]
RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1: [mAP 0.80674] [mAPH 0.8017]
RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2: [mAP 0.745899] [mAPH 0.74114]
RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1: [mAP 0.641289] [mAPH 0.63387]
RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2: [mAP 0.51159] [mAPH 0.505376]
RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_1: [mAP 0.873684] [mAPH 0.84795]
RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2: [mAP 0.839418] [mAPH 0.813563]
RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_1: [mAP 0.842307] [mAPH 0.813218]
RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2: [mAP 0.781] [mAPH 0.752629]
RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_1: [mAP 0.803274] [mAPH 0.773286]
RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2: [mAP 0.67905] [mAPH 0.651017]
RANGE_TYPE_SIGN_[0, 30)_LEVEL_1: [mAP 0] [mAPH 0]
RANGE_TYPE_SIGN_[0, 30)_LEVEL_2: [mAP 0] [mAPH 0]
RANGE_TYPE_SIGN_[30, 50)_LEVEL_1: [mAP 0] [mAPH 0]
RANGE_TYPE_SIGN_[30, 50)_LEVEL_2: [mAP 0] [mAPH 0]
RANGE_TYPE_SIGN_[50, +inf)_LEVEL_1: [mAP 0] [mAPH 0]
RANGE_TYPE_SIGN_[50, +inf)_LEVEL_2: [mAP 0] [mAPH 0]
RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_1: [mAP 0.852398] [mAPH 0.844182]
RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2: [mAP 0.847725] [mAPH 0.83955]
RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_1: [mAP 0.791685] [mAPH 0.782951]
RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2: [mAP 0.758199] [mAPH 0.749803]
RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_1: [mAP 0.674953] [mAPH 0.665672]
RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2: [mAP 0.640509] [mAPH 0.63165]
```
# Instructions

## Basic Usage
**PyTorch >= 1.9 is recommended for a better support of the checkpoint technique.**
Before using this repo, please install [TorchEx](https://github.com/Abyssaledge/TorchEx), [SpConv2](https://github.com/traveller59/spconv) (SpConv 1.x is not supported) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter).

Our implementation is based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), so just follow their [getting_started](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md) and simply run the script: `run.sh`.

**ATTENTION: It is highly recommended to check the data version if users generate data with the official MMDetection3D. MMDetection3D refactors its coordinate definition after v1.0. A hotfix is using our code to re-generate the waymo_dbinfo_train.pkl**

### Fast Waymo Evaluation:
- Copy `tools/idx2timestamp.pkl` and `tools/idx2contextname.pkl` to `./data/waymo/kitti_format/`.
- Passing the argument `--eval fast` (See `run.sh`). This argument will directly convert network outputs to Waymo `.bin` format, which is much faster than the old way.
- Follow the [installation instruction](https://github.com/Abyssaledge/faster-waymo-detection-evaluation/blob/master/docs/quick_start.md#local-compilation-without-docker-system-requirements) to install waymo evaluation tool.
- Users could further use the multi-thread Waymo evaluation tool ([link](https://github.com/Abyssaledge/waymo-open-dataset-master)) instead of the official tool for faster evaluation, following the same installation steps in the [instructions](https://github.com/Abyssaledge/faster-waymo-detection-evaluation/blob/master/docs/quick_start.md#local-compilation-without-docker-system-requirements) above (do not forget to replace the repo name).  

### For FSD:

FSD requires segmentation first, so we use an `EnableFSDDetectionHookIter` to enable the detection part after a segmentation warmup. 

If the warmup parameter is not properly modified (which is likely in your customized dataset), the memory cost might be large and the training time will be unstable (caused by CCL in CPU, we will replace it with the GPU version later).

If users do not want to waste time on the `EnableFSDDetectionHookIter`, users could first use our fast pretrain config (e.g., `fsd_sst_encoder_pretrain`) for a once-for-all warmup. The script `tools/model_converters/fsd_pretrain_converter.py` could convert the pretrain checkpoint, which can be loaded for FSD training (with a `load_from='xx'` in config). With the once-for-all pretrain, users could adopt a much short `EnableFSDDetectionHookIter`.

SST based FSD converges slower than SpConv based FSD, so we recommend users adopt the fast pretrain for SST based FSD.

### For FSDv2:
Please refer to `fsdv2_instructions.md` in `docs`.

### For SST:
It is recommended to use the configs in `./configs/sst_refactor`, which are more clear.

We only provide the single-stage model here, as for our two-stage models, please follow [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN). It's also a good choice to apply other powerful second stage detectors to our single-stage SST.

We borrow **Weighted NMS** from RangeDet and observe ~1 AP improvement on our best Vehicle model. To use it, you are supposed to clone [RangeDet](https://github.com/TuSimple/RangeDet), and simply run `pip install -v -e .` in its root directory. Then refer to `config/sst/sst_waymoD5_1x_car_8heads_wnms.py` to modify your config and enable Weight NMS. Note we only implement the CPU version for now, so it is relatively slow. Do NOT use it on 3-class models, which will lead to performance drop.

A basic config of SST with CenterHead: `./configs/sst_refactor/sst_waymoD5_1x_3class_centerhead.py`, which has significant improvement in Vehicle class.
To enable faster SSTInputLayer, clone https://github.com/Abyssaledge/TorchEx, and run `pip install -v .`.


## FSD on Argoverse 2 Dataset
### Data preprocessing
Users could use the scripts in `./tools/argo` to convert argoverse dataset into MMDet3D format.
#### Step 1. 
Download the Argoverse 2 Sensor Dataset from the [official website](https://www.argoverse.org/av2.html#download-link), and extract and organize the file into the following structure:
```
SST
├── data
│   ├── argo2
│   │   │── argo2_format
│   │   │   │   │──sensor
│   │   │   │   │   │──train
│   │   │   │   │   │   │──...
│   │   │   │   │   │──val
│   │   │   │   │   │   │──...
│   │   │   │   │   │──test
│   │   │   │   │   │   │──0c6e62d7-bdfa-3061-8d3d-03b13aa21f68
│   │   │   │   │   │   │──0f0cdd79-bc6c-35cd-9d99-7ae2fc7e165c
│   │   │   │   │   │   │──...
│   │   │── kitti_format (empty now)
```
#### Step 2.
Install [av2-api](https://github.com/argoverse/av2-api): 
```
pip install av2
```
Note there might be path and numpy version issues according to the api version, see [this](https://github.com/tusen-ai/SST/issues/92) for solution.
#### Step 3. 
Use the script `./tools/argo/argo2mmdet.py` to convert raw argo data into KITTI format. **Note I hardcode the file path in the script, while users should change these paths accordingly**.

#### Step 4.
Use  `./tools/argo/create_argo_gt_database.py` to generate GT database for CopyPaste augmentaion. **Users also need to change the hardcoded paths themself.**

#### Step 5.
Use `./tools/argo/gather_argo2_anno_feather.py` to extract validation GTs into a single file for evaluation.

#### Eventually, the directory will be organized to:
```
SST
├── data
│   ├── argo2
│   │   │── argo2_format
│   │   │   │   │──sensor
│   │   │   │   │   │──train
│   │   │   │   │   │   │──...
│   │   │   │   │   │──val
│   │   │   │   │   │   │──...
│   │   │   │   │   │──test
│   │   │   │   │   │   │──0c6e62d7-bdfa-3061-8d3d-03b13aa21f68
│   │   │   │   │   │   │──0f0cdd79-bc6c-35cd-9d99-7ae2fc7e165c
│   │   │   │   │   │   │──...
│   │   │   │   │   │──val_anno.feather (from Step 4)
│   │   │── kitti_format
│   │   │   │   │──argo2_infos_train.pkl
│   │   │   │   │──argo2_infos_val.pkl
│   │   │   │   │──argo2_infos_test.pkl
│   │   │   │   │──argo2_infos_trainval.pkl
│   │   │   │   │──training
│   │   │   │   │──testing
│   │   │   │   │──argo2_gt_database
```

### Training
We provide simple commands in `run_argo.sh`.\
For now, we adopt a segmentation pretrain in `./configs/argo/argo_onestage_12e.py`. The segmentation pretrain model can be downloaded from this [site](https://share.weiyun.com/YyJZ0fqs).
After downloading, move it to `./data/pretrain/argo_segmentation_pretrain.pth`. \
~~Note that we only support batchsize 1 per GPU for argo now, and we will update multi-sampler version very soon.~~\
We have fixed the bug when batch_size > 1 for argoverse 2 training. Users could increase the batch_size in the config.

### Inference
See `run_argo.sh`.
The pretrain weights can be downloaded from this [site](https://share.weiyun.com/YyJZ0fqs). With this weights, users could obtain our reported performance.

## Main results

We cannot distribute model weights of FSD on WOD due to the [license of WOD](https://waymo.com/open/terms). Users could contact us for the private model weights.

### FSD
WOD Validation: please refer to this [page](https://github.com/tusen-ai/SST/issues/62).

WOD Test: please refer to this [submission](https://waymo.com/open/challenges/entry/?timestamp=1665211204047769&challenge=DETECTION_3D&emailId=1cb154ab-1558)

Argoverse 2 Validation:
```
                                    AP    ATE    ASE    AOE    CDS
ARTICULATED_BUS                  0.204  0.765  0.201  0.277  0.159
BICYCLE                          0.386  0.270  0.237  0.461  0.320
BICYCLIST                        0.334  0.233  0.254  0.367  0.280
BOLLARD                          0.418  0.089  0.455  0.899  0.309
BOX_TRUCK                        0.385  0.550  0.227  0.077  0.317
BUS                              0.409  0.538  0.179  0.130  0.342
CONSTRUCTION_BARREL              0.426  0.087  0.246  0.901  0.344
CONSTRUCTION_CONE                0.412  0.090  0.432  0.911  0.307
DOG                              0.095  0.378  0.447  0.913  0.066
LARGE_VEHICLE                    0.059  0.688  0.298  0.399  0.044
MESSAGE_BOARD_TRAILER            0.000  2.000  1.000  3.142  0.000
MOBILE_PEDESTRIAN_CROSSING_SIGN  0.262  0.082  0.427  1.480  0.180
MOTORCYCLE                       0.490  0.216  0.248  0.350  0.414
MOTORCYCLIST                     0.397  0.252  0.294  0.450  0.323
PEDESTRIAN                       0.590  0.203  0.234  0.777  0.475
REGULAR_VEHICLE                  0.681  0.358  0.167  0.350  0.577
SCHOOL_BUS                       0.305  0.528  0.151  0.063  0.261
SIGN                             0.119  0.266  0.309  0.486  0.095
STOP_SIGN                        0.290  0.151  0.396  0.361  0.234
STROLLER                         0.138  0.197  0.311  0.243  0.115
TRUCK                            0.211  0.545  0.173  0.113  0.177
TRUCK_CAB                        0.148  0.837  0.292  0.150  0.110
VEHICULAR_TRAILER                0.269  0.681  0.180  0.599  0.205
WHEELCHAIR                       0.071  0.144  0.313  1.531  0.051
WHEELED_DEVICE                   0.140  0.290  0.161  0.558  0.117
WHEELED_RIDER                    0.092  0.327  0.336  0.790  0.069
AVERAGE_METRICS                  0.282  0.414  0.306  0.645  0.227
```

### SST
#### Waymo Leaderboard

|         |  #Sweeps | Veh_L1 | Ped_L1 | Cyc_L1  | Veh_L2 | Ped_L2 | Cyc_L2  | 
|---------|---------|--------|--------|---------|--------|--------|---------|
|  SST_TS_3f | 3       |  80.99  |  83.30  |  75.69   |  73.08  |  76.93  |  73.22   |

Please visit the website for detailed results: [SST_v1](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&emailId=5854f8ae-6285&timestamp=1640329826551565)

#### One stage model on Waymo validation split (refer to this [page](https://github.com/TuSimple/SST/issues/50) for the detailed performance of CenterHead SST)

|         |  #Sweeps | Veh_L1 | Ped_L1 | Cyc_L1  | Veh_L2 | Ped_L2 | Cyc_L2  | 
|---------|---------|--------|--------|---------|--------|--------|---------|
|  SST_1f | 1       |  73.57  |  80.01  |  70.72   |  64.80  |  71.66  |  68.01
|  SST_1f_center (4 SST blocks) | 1       |  75.40  |  80.28  |  71.58   |  66.76  |  72.63  |  68.89
|  SST_3f | 3       |  75.16  |  83.24  |  75.96   |  66.52  |  76.17  |  73.59   |

Note that we train the 3 classes together, so the performance above is a little bit lower than that reported in our paper.

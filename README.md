## FSD: Fully Sparse 3D Object Detection  &  SST: Single-stride Sparse Transformer 
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-pedestrian)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-pedestrian?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-cyclist)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-cyclist?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-vehicle)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle?p=embracing-single-stride-3d-object-detector)

This is the official implementation of:

[Fully Sparse 3D Object Detection](http://arxiv.org/abs/2207.10035) 
and
[Embracing Single Stride 3D Object Detector with Sparse Transformer](https://arxiv.org/pdf/2112.06375.pdf).

**🔥 FSD Preview Release**
- Code of SpConv-based FSD on Waymo is released. See `./configs/fsd/fsd_waymoD1_1x.py`
- We provide the tools for processing Argoverse 2 dataset in `./tools/argo`. We will release the instruction and configs of Argo2 model later.
- A very fast Waymo evaluation, see Usage section for detailed instructions. The whole evaluation process of FSD on Waymo costs less than **10min** with 8 2080Ti GPUs.
- We cannot distribute model weights of FSD on Waymo due to the license. Users could contact us for the private model weights.
- Before using this repo, please install [TorchEx](https://github.com/Abyssaledge/TorchEx) and SpConv2 (SpConv 1.x is not supported).

**NEWS**
- [22-09-19] The code of FSD is released here.
- [22-09-15] 🔥 FSD is accepted at NeurIPS 2022.
- [22-06-06] Support SST with CenterHead, cosine similarity in attention, faster SSTInputLayer. See Usage for details.
- 🔥 SST is accepted at CVPR 2022.
- Support Weighted NMS (CPU version) in [RangeDet](https://github.com/TuSimple/RangeDet), improving performance of vehicle class by ~1 AP.
See `Usage` section.
- We refactored the code to provide more clear function prototypes and a better understanding. See `./configs/sst_refactor`
- Supported voxel-based region partition in `./configs/sst_refactor`. Users can easily use voxel-based SST by modifying the `recover_bev` function in the backbone.
- Waymo Leaderboard results updated in [SST_v1](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&emailId=5854f8ae-6285&timestamp=1640329826551565)

<!-- **Visualization of a SST detection sequence by AB3DMOT tracking:**

![demo-min](https://user-images.githubusercontent.com/21312704/145702575-24647aed-256d-486c-835f-730584cf99ee.gif) -->


## Usage
**PyTorch >= 1.9 is recommended for a better support of the checkpoint technique.**

Our implementation is based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), so just follow their [getting_started](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md) and simply run the script: `run.sh`.

**It is highly recommended to check the data version if users generate data with the official MMDetection3D. MMDetection3D refactors its coordinate definition after v1.0. A hotfix is using our code to re-generate the waymo_dbinfo_train.pkl**

### Fast Waymo Evaluation:
- Copy `tools/idx2timestamp.pkl` and `tools/idx2contextname.pkl` to `./data/waymo/kitti_format/`.
- Passing the argument `--eval fast` (See `run.sh`). This argument will directly convert network outputs to Waymo `.bin` format, which is much faster than the old way.
- Users could further build the multi-thread Waymo evaluation tool ([link](https://github.com/Abyssaledge/waymo-open-dataset-master)) for faster evaluation. 

### For SST:
We only provide the single-stage model here, as for our two-stage models, please follow [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN). It's also a good choice to apply other powerful second stage detectors to our single-stage SST.

We borrow **Weighted NMS** from RangeDet and observe ~1 AP improvement on our best Vehicle model. To use it, you are supposed to clone [RangeDet](https://github.com/TuSimple/RangeDet), and simply run `pip install -v -e .` in its root directory. Then refer to `config/sst/sst_waymoD5_1x_car_8heads_wnms.py` to modify your config and enable Weight NMS. Note we only implement the CPU version for now, so it is relatively slow. Do NOT use it on 3-class models, which will lead to performance drop.

A basic config of SST with CenterHead: `./configs/sst_refactor/sst_waymoD5_1x_3class_centerhead.py`, which has significant improvement in Vehicle class.
To enable faster SSTInputLayer, clone https://github.com/Abyssaledge/TorchEx, and run `pip install -v .`.


## Main results
### FSD
Please refer to this [page](https://github.com/tusen-ai/SST/issues/62).

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



## Citation
Please consider citing our work as follows if it is helpful.
```
@inproceedings{fan2022embracing,
  title={{Embracing Single Stride 3D Object Detector with Sparse Transformer}},
  author={Fan, Lue and Pang, Ziqi and Zhang, Tianyuan and Wang, Yu-Xiong and Zhao, Hang and Wang, Feng and Wang, Naiyan and Zhang, Zhaoxiang},
  booktitle={CVPR},
  year={2022}
}
```
```
@article{fan2022fully,
  title={{Fully Sparse 3D Object Detection}},
  author={Fan, Lue and Wang, Feng and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2207.10035},
  year={2022}
}
```

## Acknowledgments
This project is based on the following codebases.  

* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN)

Thank the authors of CenterPoint for providing their detailed results. 

## FSD: Fully Sparse 3D Object Detection  &  SST: Single-stride Sparse Transformer 
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-pedestrian)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-pedestrian?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-cyclist)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-cyclist?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-vehicle)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle?p=embracing-single-stride-3d-object-detector)

This is the official implementation of paper:

#### Fully Sparse 3D Object Detection
[Paper Link](http://arxiv.org/abs/2207.10035)

and paper:

#### Embracing Single Stride 3D Object Detector with Sparse Transformer

[Paper Link](https://arxiv.org/pdf/2112.06375.pdf)， [中文解读](https://zhuanlan.zhihu.com/p/476056546)

**NEWS**
- [22-6-6] Support SST with CenterHead, cosine similarity in attention, faster SSTInputLayer. See Usage for details.
- 🔥 SST is accepted at CVPR 2022.
- Support Weighted NMS (CPU version) in [RangeDet](https://github.com/TuSimple/RangeDet), improving performance of vehicle class by ~1 AP.
See `Usage` section.
- We refactored the code to provide more clear function prototypes and a better understanding. See `./configs/sst_refactor`
- Supported voxel-based region partition in `./configs/sst_refactor`. Users can easily use voxel-based SST by modifying the `recover_bev` function in the backbone.
- Waymo Leaderboard results updated in [SST_v1](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&emailId=5854f8ae-6285&timestamp=1640329826551565)

**Visualization of a SST detection sequence by AB3DMOT tracking:**

![demo-min](https://user-images.githubusercontent.com/21312704/145702575-24647aed-256d-486c-835f-730584cf99ee.gif)



## Introduction
- SST is a **single-stride** network, which maintains original feature resolution from the beginning to the end of the network. Due to the characterisric of single stride, SST achieves exciting performances on small object detection (Pedestrian, Cyclist).

## Usage
**PyTorch >= 1.9 is recommended for a better support of the checkpoint technique.**
(or you can manually replace the interface of checkpoint in torch < 1.9 with the one in torch >= 1.9.)

Our implementation is based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), so just follow their [getting_started](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md) and simply run the script: `run.sh`. Then you will get a basic result of SST after 5~7 hours (depends on your devices).

#### For SST:
We only provide the single-stage model here, as for our two-stage models, please follow [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN). It's also a good choice to apply other powerful second stage detectors to our single-stage SST.

We borrow **Weighted NMS** from RangeDet and observe ~1 AP improvement on our best Vehicle model. To use it, you are supposed to clone [RangeDet](https://github.com/TuSimple/RangeDet), and simply run `pip install -v -e .` in its root directory. Then refer to `config/sst/sst_waymoD5_1x_car_8heads_wnms.py` to modify your config and enable Weight NMS. Note we only implement the CPU version for now, so it is relatively slow. Do NOT use it on 3-class models, which will lead to performance drop.

A basic config of SST with CenterHead: `./configs/sst_refactor/sst_waymoD5_1x_3class_centerhead.py`, which has significant improvement in Vehicle class.
To enable faster SSTInputLayer, clone https://github.com/Abyssaledge/TorchEx, and run `pip install -v .`.


## Main results

**_All the results of single stage models are reproducible with this repo. We also find that some improvements can usually be obtained by replacing your pillar-based conv backbone with SST.
So please let us know if you have trouble reproducing the results. Discussions are definitely welcome if you could not obtain satisfactory performances with SST in your projects._**

#### Waymo Leaderboard

|         |  #Sweeps | Veh_L1 | Ped_L1 | Cyc_L1  | Veh_L2 | Ped_L2 | Cyc_L2  | 
|---------|---------|--------|--------|---------|--------|--------|---------|
|  SST_TS_3f | 3       |  80.99  |  83.30  |  75.69   |  73.08  |  76.93  |  73.22   |

Please visit the website for detailed results: [SST_v1](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&emailId=5854f8ae-6285&timestamp=1640329826551565)

#### One stage model on Waymo validation split (refer to this [page](https://github.com/TuSimple/SST/issues/50) for detailed performanc of CenterHead SST)

|         |  #Sweeps | Veh_L1 | Ped_L1 | Cyc_L1  | Veh_L2 | Ped_L2 | Cyc_L2  | 
|---------|---------|--------|--------|---------|--------|--------|---------|
|  SST_1f | 1       |  73.57  |  80.01  |  70.72   |  64.80  |  71.66  |  68.01
|  SST_1f_center (4 SST blocks) | 1       |  75.40  |  80.28  |  71.58   |  66.76  |  72.63  |  68.89
|  SST_3f | 3       |  75.16  |  83.24  |  75.96   |  66.52  |  76.17  |  73.59   |

Note that we train the 3 classes together, so the performance above is a little bit lower than that reported in our paper.



## Citation
Please consider citing our work as follows if it is helpful.
```
@article{fan2021embracing,
  title={Embracing Single Stride 3D Object Detector with Sparse Transformer},
  author={Fan, Lue and Pang, Ziqi and Zhang, Tianyuan and Wang, Yu-Xiong and Zhao, Hang and Wang, Feng and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2112.06375},
  year={2021}
}
```

## Acknowledgments
This project is based on the following codebases.  

* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN)

Thank the authors of CenterPoint for providing their detailed results. 

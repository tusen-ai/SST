# SST: Single-stride Sparse Transformer
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-pedestrian)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-pedestrian?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-cyclist)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-cyclist?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-vehicle)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle?p=embracing-single-stride-3d-object-detector)

This is the official implementation of paper:

### Embracing Single Stride 3D Object Detector with Sparse Transformer

Authors: 
[Lue Fan](https://lue.fan/),
[Ziqi Pang](https://ziqipang.me/),
[Tianyuan Zhang](http://tianyuanzhang.com/),
[Yu-Xiong Wang](https://yxw.web.illinois.edu/),
[Hang Zhao](https://hangzhaomit.github.io/),
[Feng Wang](http://happynear.wang/),
[Naiyan Wang](https://winsty.net/),
[Zhaoxiang Zhang](https://zhaoxiangzhang.net/)

[Paper Link](https://arxiv.org/pdf/2112.06375.pdf)， [中文解读](https://zhuanlan.zhihu.com/p/476056546)

**NEWS**
- 🔥 SST is accepted at CVPR 2022.
- Support Weighted NMS (CPU version) in [RangeDet](https://github.com/TuSimple/RangeDet), improving performance of vehicle class by ~1 AP.
See `Usage` section.
- We refactored the code to provide more clear function prototypes and a better understanding. See `./configs/sst_refactor`
- Supported voxel-based region partition in `./configs/sst_refactor`. Users can easily use voxel-based SST by modifying the `recover_bev` function in the backbone.
- Waymo Leaderboard results updated in [SST_v1](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&emailId=5854f8ae-6285&timestamp=1640329826551565)

**Visualization of a sequence by AB3DMOT tracking:**

![demo-min](https://user-images.githubusercontent.com/21312704/145702575-24647aed-256d-486c-835f-730584cf99ee.gif)



## Introduction and Highlights
- SST is a **single-stride** network, which maintains original feature resolution from the beginning to the end of the network. Due to the characterisric of single stride, SST achieves exciting performances on small object detection (Pedestrian, Cyclist).
- For simplicity, except for backbone, SST is almost the same with the basic PointPillars in MMDetection3D. With such a basic setting, SST achieves state-of-the-art performance in Pedestrian and Cyclist and outperforms PointPillars more than **10 AP** only at a cost of 1.5x latency.
- SST consists of 6 **Regional Sparse Attention (SRA)** blocks, which deal with the sparse voxel set. It's similar to Submanifold Sparse Convolution (SSC), but much more powerful than SSC. It's locality and sparsity guarantee the efficiency in the single stride setting.
- The SRA can also be used in many other task to process sparse point clouds. Our implementation of SRA only relies on the pure Python APIs in PyTorch without engineering efforts
as taken in the CUDA implementation of sparse convolution. 
- **Better utilizing rich point observations.** Benefiting more from multi-sweep point clouds due to single stride. 
- Large room for further improvements. For example, **second stage, anchor-free head, IoU scores and advanced techniques from many kinds of vision transformers, etc.**

## Usage
**PyTorch >= 1.9 is recommended for a better support of the checkpoint technique.**
(or you can manually replace the interface of checkpoint in torch < 1.9 with the one in torch >= 1.9.)

Our implementation is based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), so just follow their [getting_started](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md) and simply run the script: `run.sh`. Then you will get a basic result of SST after 5~7 hours (depends on your devices).

We only provide the single-stage model here, as for our two-stage models, please follow [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN). It's also a good choice to apply other powerful second stage detectors to our single-stage SST.

We borrow **Weighted NMS** from RangeDet and observe ~1 AP improvement on our best Vehicle model. To use it, you are supposed to clone [RangeDet](https://github.com/TuSimple/RangeDet), and simply run `pip install -v -e .` in its root directory. Then refer to `config/sst/sst_waymoD5_1x_car_8heads_wnms.py` to modify your config and enable Weight NMS. Note we only implement the CPU version for now, so it is relatively slow. Do NOT use it on 3-class models, which will lead to performance drop.


## Play with your first single-stride model

In `./configs/sst/`, we provide a basic config `sst_waymoD5_1x_ped_cyc_8heads_3f` to show the power of our single-stride network on small object detection (Pedestrian and Cyclist). With this config (**only 20% training data for 12 epoch**), we can get a very good performance, which is better than most other published methods (WOD validation split):
|         |    Ped AP/APH | Cyc AP/APH  | 
|---------|--------|--------|
|  Level 1 |   80.51/75.48  |  70.44/69.43   |
|  Level 2 |   72.18/67.51  |  67.94/67.00   |

(Based on PointPillars, single stage, 3sweeps, 20% training data for 12 epochs, taking ~7 hours with 8 2080Ti GPUs)

## Main results

**_All the results of single stage models are reproducible with this repo. We also find that some improvements can usually be obtained by replacing your pillar-based conv backbone with SST.
So please let us know if you have trouble reproducing the results. Discussions are definitely welcome if you could not obtain satisfactory performances with SST in your projects._**

#### Waymo Leaderboard

|         |  #Sweeps | Veh_L1 | Ped_L1 | Cyc_L1  | Veh_L2 | Ped_L2 | Cyc_L2  | 
|---------|---------|--------|--------|---------|--------|--------|---------|
|  SST_TS_3f | 3       |  80.99  |  83.30  |  75.69   |  73.08  |  76.93  |  73.22   |

Please visit the website for detailed results: [SST_v1](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&emailId=5854f8ae-6285&timestamp=1640329826551565)

#### One stage model (based on PointPillars) on Waymo validation split

|         |  #Sweeps | Veh_L1 | Ped_L1 | Cyc_L1  | Veh_L2 | Ped_L2 | Cyc_L2  | 
|---------|---------|--------|--------|---------|--------|--------|---------|
|  SST_1f | 1       |  73.57  |  80.01  |  70.72   |  64.80  |  71.66  |  68.01
|  SST_3f | 3       |  75.16  |  83.24  |  75.96   |  66.52  |  76.17  |  73.59   |

Note that we train the 3 classes together, so the performance above is a little bit lower than that reported in our paper.



## TODO
- [ ] Build SRA block with similar API as Sparse Convolution for more convenient usage.

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

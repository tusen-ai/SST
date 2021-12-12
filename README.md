# SST: Single-stride Sparse Transformer
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

[Paper Link]() (Check again on Monday)

**Visualization of a sequence by AB3DMOT tracking:**

![demo-min](https://user-images.githubusercontent.com/21312704/145702575-24647aed-256d-486c-835f-730584cf99ee.gif)



## Introduction and Highlights
- SST is a **single-stride** network, which maintains original feature resolution from the beginning to the end of the network. Due to the characterisric of single stride, SST achieves exciting performances on small object detection (Pedestrian, Cyclist).
- For simplicity, except for backbone, SST is almost the same with the basic PointPillars in MMDetection3D. With such a basic setting, SST achieves state-of-the-art performance in Pedestrian and Cyclist and outperforms PointPillars more than **10 AP** only at a cost of 1.5x latency.
- SST consists of 6 **Regional Sparse Attention (SRA)** blocks, which deal with the sparse voxel set. It's similar to Submanifold Sparse Convolution (SSC), but much more powerful than SSC. It's locality and sparsity guarantee the efficiency in the single stride setting.
- The SRA can also be used in many other task to process sparse point clouds. Our implementation of SRA only relies on the pure Python APIs in PyTorch without engineering efforts
as taken in the CUDA implementation of sparse convolution. 
- Large room for further improvements. For example, second stage, anchor-free head, IoU scores and advanced techniques from many kinds of vision transformers, etc.

## Usage
**PyTorch >= 1.9 is recommended for a better support of the checkpoint technique.**
(or you can manually replace the interface of checkpoint in torch < 1.9 with the one in torch >= 1.9.)

Our immplementation is based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), so just follow their [getting_started](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md) and simply run the script: `run.sh`. Then you will get a basic result of SST after 5~7 hours (depends on your devices).

We only provide the single-stage model here, as for our two-stage models, please follow [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN). It's also a good choice to apply other powerful second stage detectors to our single-stage SST.

## Play with your first single-stride model

In `./configs/sst/`, we provide a basic config `sst_waymoD5_1x_ped_cyc_8heads_3f` to show the power of our single-stride network on small object detection (Pedestrian and Cyclist). With this config (**only 20% training data for 12 epoch**), we can get a very good performance, which is better than most other published methods:
|         |    Ped AP/APH | Cyc AP/APH  | 
|---------|--------|--------|
|  Level 1 |   80.51/75.48  |  70.44/69.43   |
|  Level 2 |   72.18/67.51  |  67.94/67.00   |

(20% training data, taking ~7 hours with 8 2080Ti GPUs)

## Main results

#### Single-stage Model (based on PointPillars) on Waymo validation split

|         |  #Sweeps | Veh_L1 | Ped_L1 | Cyc_L1  | 
|---------|---------|--------|--------|---------|
|  SST_1f | 1       |  73.57  |  80.01  |  70.72   |
|  SST_3f | 3       |  75.16  |  83.24  |  75.96   |

Note that we train the 3 classes together, so the performance above is a little bit lower than that reported in our paper.


## TODO
- [ ] Build SRA block with similar API as Sparse Convolution for more convenient usage.


## Acknowlegement
This project is based on the following codebases.  

* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN)

Thank the authors of CenterPoint for providing their detailed results. 

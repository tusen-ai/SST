<!-- ## FSD: Fully Sparse 3D Object Detection  &  SST: Single-stride Sparse Transformer  -->
	
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-pedestrian)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-pedestrian?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-cyclist)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-cyclist?p=embracing-single-stride-3d-object-detector)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/embracing-single-stride-3d-object-detector/3d-object-detection-on-waymo-vehicle)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-vehicle?p=embracing-single-stride-3d-object-detector) -->

### 🔥 We release the code of CTRL, the first open-sourced LiDAR-based auto-labeling system. See [ctrl_instruction](https://github.com/tusen-ai/SST/blob/main/docs/CTRL_instructions.md).
### 🔥 We release FSDv2. Better performance, easier use! Support Waymo, nuScenes, and Argoverse 2. See [fsdv2_instruction](https://github.com/tusen-ai/SST/blob/main/docs/fsdv2_instructions.md).

---

This repo contains official implementations of our series of work in LiDAR-based 3D object detection:

- [Embracing Single Stride 3D Object Detector with Sparse Transformer](https://arxiv.org/abs/2112.06375) (CVPR 2022).
- [Fully Sparse 3D Object Detection](http://arxiv.org/abs/2207.10035) (NeurIPS 2022).
- [Super Sparse 3D Object Detection](http://arxiv.org/abs/2301.02562) (TPAMI 2023).
- [Once Detected, Never Lost: Surpassing Human Performance in Offline LiDAR based 3D Object Detection](https://arxiv.org/abs/2304.12315) (ICCV 2023, Oral).
- [FSD V2: Improving Fully Sparse 3D Object Detection with Virtual Voxels](https://arxiv.org/abs/2308.03755).

Users could follow the instructions in [docs](https://github.com/tusen-ai/SST/blob/main/docs) to use this repo.


**NEWS**
- [23-08-08] The code of FSDv2 is merged into this repo.
- [23-07-14] CTRL is aceepted at ICCV 2023.
- [23-06-21] The code of FSD++ (TPAMI version of FSD) is released.
- [23-06-19] The code of CTRL is released.
- [23-03-21] The Argoverse 2 model of FSD is released. See [instructions](https://github.com/tusen-ai/SST/blob/main/instructions.md).
- [22-09-19] The code of FSD is released here.
- [22-09-15] FSD is accepted at NeurIPS 2022.
- [22-03-02] SST is accepted at CVPR 2022.
- [21-12-10] The code of SST is released.

<!-- **Visualization of a SST detection sequence by AB3DMOT tracking:**

![demo-min](https://user-images.githubusercontent.com/21312704/145702575-24647aed-256d-486c-835f-730584cf99ee.gif) -->

## Citation
Please consider citing our work as follows if it is helpful.

**Since FSD++ (TPAMI version) is accidentally excluded in Google Scholar search results, if possible, please kindly use the following bibtex**.
```
@inproceedings{fan2022embracing,
  title={{Embracing Single Stride 3D Object Detector with Sparse Transformer}},
  author={Fan, Lue and Pang, Ziqi and Zhang, Tianyuan and Wang, Yu-Xiong and Zhao, Hang and Wang, Feng and Wang, Naiyan and Zhang, Zhaoxiang},
  booktitle={CVPR},
  year={2022}
}
```
```
@inproceedings{fan2022fully,
  title={{Fully Sparse 3D Object Detection}},
  author={Fan, Lue and Wang, Feng and Wang, Naiyan and Zhang, Zhaoxiang},
  booktitle={NeurIPS},
  year={2022}
}
```
```
@article{fan2023super,
  title={Super Sparse 3D Object Detection},
  author={Fan, Lue and Yang, Yuxue and Wang, Feng and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023}
}
```
```
@inproceedings{fan2023once,
  title={Once Detected, Never Lost: Surpassing Human Performance in Offline LiDAR based 3D Object Detection},
  author={Fan, Lue and Yang, Yuxue and Mao, Yiming and Wang, Feng and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  booktitle={ICCV},
  year={2023}
}
```
```
@article{fan2023fsdv2,
  title={FSD V2: Improving Fully Sparse 3D Object Detection with Virtual Voxels},
  author={Fan, Lue and Wang, Feng and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2308.03755},
  year={2023}
}
```

## Acknowledgments
This project is based on the following codebases.  

* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [LiDAR-RCNN](https://github.com/TuSimple/LiDAR_RCNN)

Thank the authors of CenterPoint for providing their detailed results. 

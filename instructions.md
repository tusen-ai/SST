# Instructions


## FSD on Argoverse 2 Dataset
### Data preprocessing
Users could use the scripts in `./tools/argo` to convert argoverse dataset into MMDet3D format. Detailed instructions will be followed very soon.

### Training
We provide simple commands in `run_argo.sh`.\
For now, we adopt a segmentation pretrain in `./configs/argo/argo_onestage_12e.py`. The segmentation pretrain model can be downloaded from this [site](https://share.weiyun.com/YyJZ0fqs).
After downloading, move it to `./data/pretrain/argo_segmentation_pretrain.pth`. \
Note that we only support batchsize 1 per GPU for argo now, and we will update multi-sampler version very soon.

### Inference
See `run_argo.sh`.
The pretrain weights can be downloaded from this [site](https://share.weiyun.com/YyJZ0fqs). With this weights, users could obtain our reported performance:
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

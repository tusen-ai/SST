#!/bin/sh
pip install numpy==1.21
bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py $GPUS_PER_NODE --work-dir /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/$JOB_ID --cfg-options evaluation.pklfile_prefix=/cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/$JOB_ID/results evaluation.metric=nuscenes

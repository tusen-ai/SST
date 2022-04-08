CONFIG=sst_nuscenesD5_1x_3class_8heads_v2
bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py 1 --work-dir ./work_dirs/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./work_dirs/$CONFIG/results evaluation.metric=nuscenes

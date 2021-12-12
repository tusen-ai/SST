CONFIG=sst_waymoD5_1x_ped_cyc_8heads_3f
bash tools/dist_train.sh configs/sst/$CONFIG.py 8 --work-dir ./work_dirs/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./work_dirs/$CONFIG/results evaluation.metric=waymo

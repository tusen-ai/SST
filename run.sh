CONFIG=sst_nuscenes_masked
bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py 8 --work-dir ./work_dirs/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./work_dirs/$CONFIG/results evaluation.metric=nuscenes

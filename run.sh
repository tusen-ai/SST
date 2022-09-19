DIR=fsd
WORK=work_dirs
CONFIG=fsd_waymoD1_1x
bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/retrain_results evaluation.metric=waymo --seed 1


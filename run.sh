DIR=fsd
WORK=work_dirs
CONFIG=fsd_waymoD1_1x
bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/results evaluation.metric=fast --seed 1

# CTRL training
# DIR=ctrl
# CONFIG=ctrl_veh_24e
# bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --no-validate

# fast Waymo Evaluation, for all waymo-based models
# bash tools/dist_test.sh configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/latest.pth 8 --options "pklfile_prefix=./$WORK/$CONFIG/results" --eval fast

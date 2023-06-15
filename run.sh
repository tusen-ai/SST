DIR=ctrl
WORK=work_dirs
CONFIG=ctrl_veh_4x
# bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/results evaluation.metric=fast --seed 1

# fast Waymo Evaluation
bash tools/dist_test.sh configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/epoch_24.pth 8 --options "pklfile_prefix=./$WORK/$CONFIG/release_debug_results" --eval waymo

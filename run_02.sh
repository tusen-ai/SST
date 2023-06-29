DIR=fsdv2
WORK=work_dirs
CONFIG=fsdv2_waymo_1x
#CONFIG=fsdv2_baseline_debug
#bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.jsonfile_prefix=./$WORK/$CONFIG/results evaluation.metric=bbox --seed 1
bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/results evaluation.metric=fast --seed 1

#  sleep 60
# CONFIG=fsdv2_nus_2x_bs2_large
# bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.jsonfile_prefix=./$WORK/$CONFIG/results evaluation.metric=bbox --seed 1
# bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/  --seed 1 --no-validate

# sleep 60
# CONFIG=fsdv2_argo_1x_centroid_scalegt01_l1
# bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/results evaluation.metric=fast --seed 1

# fast Waymo Evaluation
#bash tools/dist_test.sh configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/epoch_24.pth 8 --options "pklfile_prefix=./$WORK/$CONFIG/results" --eval fast
#bash tools/dist_test.sh configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/latest.pth 8 --options "jsonfile_prefix=./$WORK/$CONFIG/results" --eval bbox

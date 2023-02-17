DIR=fsd
WORK=work_dirs
CONFIG=fsd_9f_no_cp
bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/results evaluation.metric=fast --seed 1

# fast Waymo Evaluation
# bash tools/dist_test.sh configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/latest.pth 8 --options "pklfile_prefix=./$WORK/$CONFIG/results_mini_3rots_doubleflip_wnms_meanscore_mthr04" --eval fast

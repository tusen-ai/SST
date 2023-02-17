DIR=fsd
WORK=work_dirs
CONFIG=fsd_waymoD1_1x_futuresweeps_framedrop
#bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/results evaluation.metric=fast --seed 1

# fast Waymo Evaluation
bash tools/dist_test.sh configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/lastest.pth 8 --options "pklfile_prefix=./$WORK/$CONFIG/results_mini_doubleflip" --eval fast

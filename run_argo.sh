DIR=argo2
WORK=work_dirs
CONFIG=argo_onestage_12e
bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/results evaluation.metric=fast

# for segmentation pretrain
# CONFIG=argo_segmentation_pretrain
# bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --no-validate

# bash tools/dist_test.sh configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/latest.pth 8 --options "pklfile_prefix=./$WORK/$CONFIG/results" --eval fast

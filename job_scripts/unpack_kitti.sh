#!/usr/bin/env bash
GPU_TYPE=${GPU_TYPE:-A40}
src_folder=/mimer/NOBACKUP/Datasets/KITTI-Vision-Benchmark-Suite/object
gt_db_folder=/mimer/NOBACKUP/groups/snic2021-7-127/eliassv/data/kitti/
trg_folder=$TMPDIR/SST_$GPU_TYPE/data/kitti
mkdir $trg_folder
mkdir $trg_folder/ImageSets

unzip -q -d $trg_folder $src_folder/data_object_calib.zip &
unzip -q -d $trg_folder $src_folder/data_object_image_2.zip &
unzip -q -d $trg_folder $src_folder/data_object_label_2.zip &
unzip -q -d $trg_folder $src_folder/data_object_velodyne.zip &
wait

wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O $trg_folder/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O $trg_folder/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O $trg_folder/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O $trg_folder/ImageSets/trainval.txt

ln -s $gt_db_folder/kitti_dbinfos_train.pkl $trg_folder/kitti_dbinfos_train.pkl
ln -s $gt_db_folder/kitti_gt_database $trg_folder/kitti_gt_database
ln -s $gt_db_folder/kitti_infos_test_mono3d.coco.json $trg_folder/kitti_infos_test_mono3d.coco.json
ln -s $gt_db_folder/kitti_infos_test.pkl $trg_folder/kitti_infos_test.pkl
ln -s $gt_db_folder/kitti_infos_train_mono3d.coco.json $trg_folder/kitti_infos_train_mono3d.coco.json
ln -s $gt_db_folder/kitti_infos_train.pkl $trg_folder/kitti_infos_train.pkl
ln -s $gt_db_folder/kitti_infos_trainval_mono3d.coco.json $trg_folder/kitti_infos_trainval_mono3d.coco.json
ln -s $gt_db_folder/kitti_infos_trainval.pkl $trg_folder/kitti_infos_trainval.pkl
ln -s $gt_db_folder/kitti_infos_val_mono3d.coco.json $trg_folder/kitti_infos_val_mono3d.coco.json
ln -s $gt_db_folder/kitti_infos_val.pkl $trg_folder/kitti_infos_val.pkl
ln -s $gt_db_folder/training/velodyne_reduced $trg_folder/training/velodyne_reduced
ln -s $gt_db_folder/testing/velodyne_reduced $trg_folder/testing/velodyne_reduced
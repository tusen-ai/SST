#!/usr/bin/env bash
GPU_TYPE=A40
src_folder=/mimer/NOBACKUP/Datasets/NuScenes_v1.0/Trainval-Boston
trg_folder=$TMPDIR/SST_$GPU_TYPE/data/nuscenes

#cp -R $src_folder/Metadata/maps $trg_folder &
#cp -R $src_folder/Metadata/v1.0-trainval $trg_folder &
#cp -R $src_folder/trainval01_blobs/*/LIDAR_TOP/* $trg_folder &
# ...
# Script not done. Using unpack_trainval.sh 

tar -xf $src_folder/Metadata/v1.0-trainval_meta_Boston.tgz -C $trg_folder &
tar -xf $src_folder/v1.0-trainval01_blobs.tgz -C $trg_folder */LIDAR_TOP/* &
tar -xf $src_folder/v1.0-trainval02_blobs.tgz -C $trg_folder */LIDAR_TOP/* &
tar -xf $src_folder/v1.0-trainval03_blobs.tgz -C $trg_folder */LIDAR_TOP/* &
tar -xf $src_folder/v1.0-trainval04_blobs.tgz -C $trg_folder */LIDAR_TOP/* &
tar -xf $src_folder/v1.0-trainval05_blobs.tgz -C $trg_folder */LIDAR_TOP/* &
tar -xf $src_folder/v1.0-trainval06_blobs.tgz -C $trg_folder */LIDAR_TOP/* &
tar -xf $src_folder/v1.0-trainval07_blobs.tgz -C $trg_folder */LIDAR_TOP/* &
tar -xf $src_folder/v1.0-trainval08_blobs.tgz -C $trg_folder */LIDAR_TOP/* &
tar -xf $src_folder/v1.0-trainval09_blobs.tgz -C $trg_folder */LIDAR_TOP/* &
tar -xf $src_folder/v1.0-trainval10_blobs.tgz -C $trg_folder */LIDAR_TOP/* &
wait

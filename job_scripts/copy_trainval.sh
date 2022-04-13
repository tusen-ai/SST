#!/usr/bin/env bash
GPU_TYPE=A40
src_folder=/mimer/NOBACKUP/Datasets/NuScenes_v1.0/Trainval-Boston
trg_folder=$TMPDIR/SST_$GPU_TYPE/data/nuscenes

#cp -R $src_folder/Metadata/maps $trg_folder &
#cp -R $src_folder/Metadata/v1.0-trainval $trg_folder &
#cp -R $src_folder/trainval01_blobs/*/LIDAR_TOP/* $trg_folder &
# ...
# Script not done. Using unpack_trainval.sh 

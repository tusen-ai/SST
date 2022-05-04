#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-127 -p alvis
#SBATCH -t 16:00:00
#SBATCH --gpus-per-node=A40:4
#SBATCH -N 1
#SBATCH -J "Create database"  # multi node, multi GPU
CONFIG=${1:-sst_nuscenesD5_1x_3class_8heads_v2}
REPO_NUMBER=${2:-1}  # Choose between repo 1 or 2
echo $CONFIG
GPUS_PER_NODE=4
export GPU_TYPE=A40
# Options for GPU type are T4 or A40. Choose A40 also when running on V100, A100 or A100fat.
export OMP_NUM_THREADS=16  # cores per gpu (16 cores per A40)


echo $HOSTNAME
echo $SLURM_JOB_NODELIST

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=12345
export PORT=$MASTER_PORT
export JOB_ID=$SLURM_JOB_ID
export NGPUS_PER_NODE=$(echo "$SLURM_GPUS_PER_NODE" | sed 's/[A-Z0-9]*:\([0-9]*\)*/\1/')

echo ""
echo "This job $JOB_ID was started as:
  bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py $GPUS_PER_NODE \
    --work-dir /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/$JOB_ID \
    --cfg-options evaluation.metric=nuscenes" ${@:3}
echo ""

echo ""
echo "Start copying repo to '$TMPDIR'"
if [ $REPO_NUMBER == 1 ]
then
   echo "Taking SST_${GPU_TYPE}"
   cp -r /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/SST_${GPU_TYPE}/ $TMPDIR/SST_${GPU_TYPE}
else
   echo "Taking SST_${GPU_TYPE}_2"
   cp -r /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/SST_${GPU_TYPE}_2/ $TMPDIR/SST_${GPU_TYPE}
fi
echo ""
echo "Copying of repo to tempdir is now done."
echo ""
cd $TMPDIR/SST_$GPU_TYPE
mkdir data
mkdir data/nuscenes
echo ""
echo "Start copying nusenes info to '$TMPDIR/SST_$GPU_TYPE/'"
# unzip -q -d $TMPDIR/SST_$GPU_TYPE /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/nuscenes_info/nuscenes_infos_v2.zip
echo ""
echo "Copying of nusenes info to repo in tempdir is now done."
echo ""

src_folder=/mimer/NOBACKUP/Datasets/NuScenes_v1.0/Trainval-Boston
src_folder_test=/mimer/NOBACKUP/Datasets/NuScenes_v1.0/Test
trg_folder=$TMPDIR/SST_$GPU_TYPE/data/nuscenes

tar -xf $src_folder/Metadata/v1.0-trainval_meta_Boston.tgz -C $trg_folder &
tar -xf $src_folder_test/Metadata_Boston/v1.0-test_meta.tgz -C $trg_folder &
wait

tar -xf $src_folder/v1.0-trainval01_blobs.tgz -C $trg_folder &
tar -xf $src_folder/v1.0-trainval02_blobs.tgz -C $trg_folder &
tar -xf $src_folder/v1.0-trainval03_blobs.tgz -C $trg_folder &
tar -xf $src_folder/v1.0-trainval04_blobs.tgz -C $trg_folder &
tar -xf $src_folder/v1.0-trainval05_blobs.tgz -C $trg_folder &
tar -xf $src_folder/v1.0-trainval06_blobs.tgz -C $trg_folder &
tar -xf $src_folder/v1.0-trainval07_blobs.tgz -C $trg_folder &
tar -xf $src_folder/v1.0-trainval08_blobs.tgz -C $trg_folder &
tar -xf $src_folder/v1.0-trainval09_blobs.tgz -C $trg_folder &
tar -xf $src_folder/v1.0-trainval10_blobs.tgz -C $trg_folder &
tar -xf $src_folder_test/v1.0-test_blobs-Boston.tgz -C $trg_folder &
wait
echo ""
echo "Start copying dataset to '$TMPDIR/SST_$GPU_TYPE/'"

echo ""
echo "Copying of dataset to repo in tempdir is now done."
echo ""

cd $TMPDIR/SST_$GPU_TYPE
singularity exec --pwd $TMPDIR/SST_$GPU_TYPE \
  /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/sst_env/mmdetection3d_$GPU_TYPE.sif \
  python tools/create_data.py nuscenes --root-path=./data/nuscenes --max-sweeps=10 --out-dir=./data/nuscenes --extra-tag=nuscenes
# ${@:3} grabs everything after the third argument these are used to overwrite settings in the config file
# e.g. load_from="config/path/epoch_n.pth" loads the weights from the provided model
mv data/nuscenes/samples ../samples
mv data/nuscenes/sweeps ../sweeps
mv data/nuscenes/maps ../maps
mv data/nuscenes/v1.0-trainval ../v1.0-trainval
mv data/nuscenes/v1.0-test ../v1.0-test
zip -q -r ~/nuscenes_infos/nuscenes_infos_v3.zip data

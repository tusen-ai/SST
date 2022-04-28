#!/usr/bin/env bash
#SBATCH -t 0-16:00:00
#SBATCH --gpus=6
#SBATCH -N 1
#SBATCH --output=/proj/deep-mot/eliassv/slurm-out/slurm-%j.out
#SBATCH -J "Some job name"  # single node, multi GPU
CONFIG=${1:-sst_nuscenesD5_1x_3class_8heads_v2}
REPO_NUMBER=${2:-1}  # Choose between repo 1 or 2
echo $CONFIG
GPUS_PER_NODE=6  # Max is 8 per mode
export OMP_NUM_THREADS=16  # cores per gpu (16 cores per A40)

echo $HOSTNAME
echo $SLURM_JOB_NODELIST

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=12345
export PORT=$MASTER_PORT
export JOB_ID=$SLURM_JOB_ID

echo ""
echo "This job $JOB_ID was started as:
  bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py $GPUS_PER_NODE \
    --work-dir /proj/deep-mot/eliassv/jobs/$JOB_ID \
    --cfg-options evaluation.metric=nuscenes" ${@:3}
echo ""

singularity exec --nv --pwd /proj/deep-mot/eliassv/SST_${REPO_NUMBER} \
  /proj/deep-mot/eliassv/sst_env/mmdetection3d_A40.sif \
  bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py $GPUS_PER_NODE \
  --work-dir /proj/deep-mot/eliassv/jobs/$JOB_ID \
  --cfg-options evaluation.metric=nuscenes ${@:3}
# ${@:3} grabs everything after the third argument these are used to overwrite settings in the config file
# e.g. load_from="config/path/epoch_n.pth" loads the weights from the provided model

cp /proj/deep-mot/eliassv/slurm-out/slurm-$JOB_ID.out /proj/deep-mot/eliassv/jobs/$JOB_ID

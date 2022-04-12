#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-127 -p alvis
#SBATCH -t 03:10:00
#SBATCH --gpus-per-node=A40:4
#SBATCH -N 1
#SBATCH --output=/cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/slurm-out/slurm-%j.out
#SBATCH -J "MNMG PyTorch"  # multi node, multi GPU
CONFIG=sst_nuscenesD5_1x_3class_8heads_v2
GPUS_PER_NODE=4
GPU_TYPE=A40

echo $HOSTNAME
echo $SLURM_JOB_NODELIST

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=12345
export JOB_ID=$SLURM_JOB_ID
export NGPUS_PER_NODE=$(echo "$SLURM_GPUS_PER_NODE" | sed 's/[A-Z0-9]*:\([0-9]*\)*/\1/')

echo ""
echo "This job $JOB_ID was started as: --dataroot $TMPDIR/SST_$GPU_TYPE/data/nuscenes
 --storage-dir /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/$JOB_ID --dataset=v1.0-mini $@"
echo ""

echo ""
echo "Start copying repo to '$TMPDIR'"
cp -r /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/SST_$GPU_TYPE/ $TMPDIR/
echo ""
echo "Copying of repo to tempdir is now done."
echo ""

######################## DO NOT EXIST YET ##################################################
echo ""
echo "Start copying nusenes info to '$TMPDIR/SST_$GPU_TYPE/'"
tar -xf /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/nuscenes_info/nuscenes_mini_infos.zip -C $TMPDIR/SST_$GPU_TYPE
echo ""
echo "Copying of nusenes info to repo in tempdir is now done."
echo ""

######################## DO NOT EXIST YET ##################################################
echo ""
echo "Start copying dataset to '$TMPDIR/SST_$GPU_TYPE/'"
job_scripts_path=$(dirname ${BASH_SOURCE[0]})
source $job_scripts_path//copy_mini.sh
echo ""
echo "Copying of dataset to repo in tempdir is now done."
echo ""

singularity exec --pwd $TMPDIR/ /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/sst_env/mmdetection3d_avlis3.sif bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py $GPUS_PER_NODE $MASTER_PORT --work-dir /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/$JOB_ID --cfg-options evaluation.pklfile_prefix=/cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/$JOB_ID/results evaluation.metric=nuscenes

cp /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/slurm-out/slurm-$JOB_ID.out /cephyr/NOBACKUP/groups/snic2021-7-127/eliassv/jobs/$JOB_ID

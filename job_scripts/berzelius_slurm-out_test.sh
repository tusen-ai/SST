
#!/usr/bin/env bash
#SBATCH -t 0-16:00:00
#SBATCH --gpus=8
#SBATCH -N 1
#SBATCH --output=/proj/deep-mot/eliassv/slurm-out/slurm-%j.out
#SBATCH -J "Some job name"  # single node, multi GPU
CONFIG=${1:-sst_nuscenesD5_1x_3class_8heads_v2}
REPO_NUMBER=${2:-1}  # Choose between repo 1 or 2
echo $CONFIG
GPUS_PER_NODE=8
export GPU_TYPE=A40
export OMP_NUM_THREADS=16  # cores per gpu (16 cores per A100 on berzelius)

echo "hostname: $HOSTNAME"
echo "slurm_job_nodelist: $SLURM_JOB_NODELIST"

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=12345
export PORT=$MASTER_PORT
export JOB_ID=$SLURM_JOB_ID
export NGPUS_PER_NODE=$(echo "$SLURM_GPUS_PER_NODE" | sed 's/[A-Z0-9]*:\([0-9]*\)*/\1/')

echo "slurm_job_id $SLURM_JOB_ID"
echo "ngpus_per_node $NGPUS_PER_NODE"
echo "slurm_gpus_per_node $SLURM_GPUS_PER_NODE"

echo ""
echo "The slurm-out test is done."
echo ""


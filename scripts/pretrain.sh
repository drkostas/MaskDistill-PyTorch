#!/bin/bash -l
#SBATCH -J md-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH -A YOUR_ACCOUNT  # Set to your SLURM account
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=logs/pretrain/%x/%j.log
#SBATCH --signal=SIGUSR1@90

# MaskDistill pretraining: ViT-Base, 300 epochs, 4 GPUs
# Usage: sbatch scripts/pretrain.sh [config_path]
#   Default: configs/pretrain.yaml (block masking 40%, dense encoding)

CONFIG=${1:-configs/pretrain.yaml}

if [ -f .env ]; then source .env; fi
nvidia-smi
echo "Job ID: ${SLURM_JOB_ID}, Host: $(hostname), Config: ${CONFIG}"

cd "${SLURM_SUBMIT_DIR}" || exit 1
# module load cuda  # Uncomment and adjust for your cluster
# module load cudnn  # Uncomment and adjust for your cluster
source .venv/bin/activate

unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Run started at: $(date)"

srun torchrun \
  --nproc_per_node=${SLURM_GPUS_PER_NODE} \
  --nnodes=${SLURM_NNODES} \
  --node_rank=${SLURM_NODEID} \
  --rdzv-id=${SLURM_JOB_ID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  -m src.train \
  --cfg "${CONFIG}"

echo "################################################################"
echo "Run completed at: $(date)"
echo "################################################################"

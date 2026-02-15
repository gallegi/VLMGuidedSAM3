#!/bin/bash
#SBATCH -J sam3_vlm_rl                          # Job name
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=32                       # CPU cores (more for Ray workers + data loading)
#SBATCH --nodes=1                                # Single node
#SBATCH --partition=h100-ferranti                # H100 partition
#SBATCH --time=0-72:00                           # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:4                             # 4x H100 80GB
#SBATCH --mem=256GB                              # Total memory
#SBATCH --output=/weka/lensch/lhr294/SAM3_VLM/logs/sam3_vlm_rl-%j.out
#SBATCH --error=/weka/lensch/lhr294/SAM3_VLM/logs/sam3_vlm_rl-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=valay.bundele@uni-tuebingen.de

set -x

# Diagnostic
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi

# ============================================================
# Source conda
# ============================================================
source ~/.bashrc
source /weka/cweiss/miniconda3/etc/profile.d/conda.sh
conda activate /weka/lensch/lhr294/easyr1_env

# ============================================================
# Environment variables
# ============================================================
export MALLOC_TRIM_THRESHOLD_=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# Ray / vLLM
export DECORD_EOF_RETRY_MAX=2048001
export WANDB_API_KEY=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

# ============================================================
# Model selection (EDIT THIS)
# ============================================================
# Options: "2b" or "4b"
MODEL_SIZE="4b"  # Change to "2b" to use Qwen3-VL-2B-Instruct

# ============================================================
# Paths  (EDIT THESE for your server layout)
# ============================================================
EASYR1_DIR="/weka/lensch/lhr294/SAM3_VLM/OneThinker/EasyR1"
export PYTHONPATH="${EASYR1_DIR}:${PYTHONPATH}"

# Set model path and hyperparameters based on model size
if [ "${MODEL_SIZE}" = "2b" ]; then
    MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
    TP_SIZE=2            # Tensor parallel size for 2B model (can use 1 or 2)
    ROLLOUT_BS=256       # Can use larger batch size for 2B on H100
    GLOBAL_BS=64         # Can use larger batch size for 2B on H100
else
    MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
    TP_SIZE=4            # Tensor parallel size for 4B model
    ROLLOUT_BS=128       # Rollout batch size
    GLOBAL_BS=32         # Actor global batch size
fi

# Training data
TRAIN_FILE="/weka/lensch/lhr294/SAM3_VLM/OneThinker/sam3_tracking_data/tracking_verification_train.json"
IMAGE_DIR="/weka/lensch/lhr294/SAM3_VLM/OneThinker/sam3_tracking_data/images"

# Checkpoints
CKPT_DIR="/weka/lensch/lhr294/SAM3_VLM/OneThinker/checkpoints/tracking_verification_ema_grpo"

# Reward function (absolute path so Ray workers find it)
REWARD_FUNCTION="${EASYR1_DIR}/verl/reward_function/tracking_verification_reward.py:compute_score"

# ============================================================
# Training hyperparameters (scaled for 4x H100 80GB)
# ============================================================
project_name='sam3_vlm_tracking'
exp_name='tracking_verification_ema_grpo'

MB_PER_UPDATE=2      # Micro batch per device for update (H100 has room)
MB_PER_EXP=2         # Micro batch per device for experience
N_GPUS_PER_NODE=4
NNODES=1

# --- LoRA config ---
LORA_RANK=64         # Set to 0 to disable LoRA
LORA_LR=1e-5         # Higher LR for LoRA

# Create directories
mkdir -p /weka/lensch/lhr294/SAM3_VLM/logs
mkdir -p "${CKPT_DIR}"

cd "${EASYR1_DIR}"

# ============================================================
# Launch training
# ============================================================
python3 -m verl.trainer.main \
    config=examples/tracking_verification_ema_grpo.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TRAIN_FILE}" \
    data.image_dir="${IMAGE_DIR}" \
    worker.reward.reward_function="${REWARD_FUNCTION}" \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.lora.rank="${LORA_RANK}" \
    worker.actor.optim.lr="${LORA_LR}" \
    worker.actor.padding_free=true \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.offload.offload_optimizer=false \
    worker.rollout.n=8 \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.rollout.enforce_eager=false \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    algorithm.online_filtering=true \
    algorithm.filter_key=accuracy \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=100 \
    trainer.save_checkpoint_path="${CKPT_DIR}"

#!/usr/bin/env bash
set -x

# Set PYTHONPATH to include EasyR1 directory so verl module can be found
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${EASYR1_DIR}:${PYTHONPATH}"

export DECORD_EOF_RETRY_MAX=2048001
export WANDB_API_KEY=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

# ============================================================
# Tracking verification RL training with LoRA
# Finetunes Qwen3-VL-2B or 4B to verify SAM3 tracking
# predictions and produce correction bboxes when wrong.
# Uses LoRA for parameter-efficient finetuning.
# ============================================================

project_name='sam3_vlm_tracking'
exp_name='tracking_verification_ema_grpo'

# --- Model selection (EDIT THIS) ---
# Options: "2b" or "4b"
MODEL_SIZE="2b"  # Change to "2b" to use Qwen3-VL-2B-Instruct

# --- Paths (EDIT THESE) ---
if [ "${MODEL_SIZE}" = "2b" ]; then
    MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
    TP_SIZE=2            # Tensor parallel size for 2B model (can use 1 or 2)
    ROLLOUT_BS=16        # Can use larger batch size for 2B
    GLOBAL_BS=8         # Can use larger batch size for 2B
else
    MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
    TP_SIZE=4            # Tensor parallel size for 4B model
    ROLLOUT_BS=16        # Rollout batch size (reduced for 4x 24GB GPUs)
    GLOBAL_BS=8          # Actor global batch size (reduced)
fi

# Training data JSON produced by gen_tracking_verification_data.py
TRAIN_FILE="/graphics/scratch3/datasets/sav_train/tracking_verification_data/tracking_verification_train.json"

# Base directory: image paths in TRAIN_FILE are relative to this
IMAGE_DIR="/graphics/scratch3/datasets/sav_train/tracking_verification_data/images"

# --- Training hyperparameters ---
MB_PER_UPDATE=1      # Micro batch per device for update
MB_PER_EXP=1         # Micro batch per device for experience
N_GPUS_PER_NODE=4
NNODES=1

# --- LoRA config ---
LORA_RANK=64         # Set to 0 to disable LoRA
LORA_LR=1e-5         # Higher LR for LoRA

# Change to EasyR1 directory for config paths to work
cd "${EASYR1_DIR}"

# Get absolute path to reward function
REWARD_FUNCTION="${EASYR1_DIR}/verl/reward_function/tracking_verification_reward.py:compute_score"

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
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    algorithm.online_filtering=true \
    algorithm.filter_key=accuracy \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=100 \
    trainer.save_checkpoint_path=checkpoints/${exp_name}
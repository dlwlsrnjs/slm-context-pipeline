#!/bin/bash
# Curriculum training for ICRL-Math: Qwen2.5-3B-Instruct, 3 -> 2 -> 0 shot.
# Skips the 1-shot stage per ICRL ablation (3-stage > 4-stage).
#
# Prerequisites (run install_into_icrl.sh first):
#   - ICRL repo patched with our math_fewshot reward
#   - Python sandbox running at http://127.0.0.1:8000/retrieve
#   - 3/2/0-shot math parquet under data/math_{3,2,0}shot
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ICRL_DIR=/home/jklee/ondevice/ICRL \
#     bash train_curriculum_math.sh

set -e

ICRL_DIR="${ICRL_DIR:-/home/jklee/ondevice/ICRL}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- Environment ----------
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export N_GPUS="${N_GPUS:-4}"
export PYTHONPATH="$ICRL_DIR:$PYTHONPATH"

# ---------- Config ----------
WAND_PROJECT='ICRL-MATH-Curriculum'
BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HERE/checkpoints}"
STAGE1_DATA_DIR="${STAGE1_DATA_DIR:-$HERE/data/math_3shot}"
STAGE2_DATA_DIR="${STAGE2_DATA_DIR:-$HERE/data/math_2shot}"
STAGE3_DATA_DIR="${STAGE3_DATA_DIR:-$HERE/data/math_0shot}"
SANDBOX_URL="${SANDBOX_URL:-http://127.0.0.1:8000/retrieve}"

STAGE1_STEPS="${STAGE1_STEPS:-100}"
STAGE2_STEPS="${STAGE2_STEPS:-100}"
STAGE3_STEPS="${STAGE3_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-50}"

# 3B model — slightly tighter than ICRL's 7B defaults to fit memory.
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=2048
MAX_START_LENGTH=3500
MAX_OBS_LENGTH=500          # sandbox stdout cap
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE=8
N_AGENT_ROLLOUTS=8          # ICRL uses 4 for 7B; 8 for richer advantage estimate on 3B
TEMPERATURE=1.0
ACCURACY_WEIGHT=0.8
FORMAT_WEIGHT=0.2

# ---------- Training stage runner ----------
run_training_stage() {
    local STAGE=$1
    local NUM_SHOTS=$2
    local TOTAL_STEPS=$3
    local MODEL_PATH=$4
    local EXPERIMENT_NAME=$5
    local DATA_DIR=$6

    echo "=========================================="
    echo " Stage $STAGE: ${NUM_SHOTS}-shot"
    echo "  model  = $MODEL_PATH"
    echo "  data   = $DATA_DIR"
    echo "  steps  = $TOTAL_STEPS"
    echo "  exp    = $EXPERIMENT_NAME"
    echo "=========================================="

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_fewshot \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.train_data_num=null \
        data.val_data_num=256 \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.val_batch_size=32 \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.max_start_length=$MAX_START_LENGTH \
        data.max_obs_length=$MAX_OBS_LENGTH \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
        actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
        actor_rollout_ref.actor.fsdp_config.param_offload=true \
        actor_rollout_ref.actor.fsdp_config.grad_offload=true \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        algorithm.no_think_rl=false \
        actor_rollout_ref.rollout.n_agent=$N_AGENT_ROLLOUTS \
        actor_rollout_ref.rollout.temperature=$TEMPERATURE \
        actor_rollout_ref.actor.state_masking=true \
        trainer.logger=['wandb'] \
        +trainer.val_only=false \
        +trainer.val_before_train=true \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=10 \
        trainer.project_name=$WAND_PROJECT \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.total_epochs=1 \
        trainer.total_training_steps=$TOTAL_STEPS \
        trainer.default_local_dir=$CHECKPOINT_DIR/$EXPERIMENT_NAME \
        max_turns=6 \
        retriever.url=$SANDBOX_URL \
        retriever.topk=1 \
        +reward.type=fewshot \
        +reward.accuracy_weight=$ACCURACY_WEIGHT \
        +reward.format_weight=$FORMAT_WEIGHT \
        2>&1 | tee ${EXPERIMENT_NAME}.log

    echo "Stage $STAGE done."
    echo
}

# ---------- Stage 1: 3-shot ----------
STAGE1_EXPERIMENT="icrl-math-stage1-3shot-qwen2.5-3b"
run_training_stage 1 3 $STAGE1_STEPS "$BASE_MODEL" "$STAGE1_EXPERIMENT" "$STAGE1_DATA_DIR"

STAGE1_CKPT="$CHECKPOINT_DIR/$STAGE1_EXPERIMENT/actor/global_step_${STAGE1_STEPS}"
if [ ! -d "$STAGE1_CKPT" ]; then
    STAGE1_CKPT=$(ls -d $CHECKPOINT_DIR/$STAGE1_EXPERIMENT/actor/global_step_* 2>/dev/null | sort -V | tail -1)
fi
echo "Stage 1 ckpt: $STAGE1_CKPT"

# ---------- Stage 2: 2-shot ----------
STAGE2_EXPERIMENT="icrl-math-stage2-2shot-qwen2.5-3b"
run_training_stage 2 2 $STAGE2_STEPS "$STAGE1_CKPT" "$STAGE2_EXPERIMENT" "$STAGE2_DATA_DIR"

STAGE2_CKPT="$CHECKPOINT_DIR/$STAGE2_EXPERIMENT/actor/global_step_${STAGE2_STEPS}"
if [ ! -d "$STAGE2_CKPT" ]; then
    STAGE2_CKPT=$(ls -d $CHECKPOINT_DIR/$STAGE2_EXPERIMENT/actor/global_step_* 2>/dev/null | sort -V | tail -1)
fi
echo "Stage 2 ckpt: $STAGE2_CKPT"

# ---------- Stage 3: 0-shot  (skipping 1-shot per ICRL ablation) ----------
STAGE3_EXPERIMENT="icrl-math-stage3-0shot-qwen2.5-3b"
run_training_stage 3 0 $STAGE3_STEPS "$STAGE2_CKPT" "$STAGE3_EXPERIMENT" "$STAGE3_DATA_DIR"

echo "=========================================="
echo " All stages done."
echo " Final model: $CHECKPOINT_DIR/$STAGE3_EXPERIMENT/actor/global_step_${STAGE3_STEPS}"
echo "=========================================="

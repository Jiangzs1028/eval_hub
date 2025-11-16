#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODELS=(
  /mnt/data/kw/jzs/OT-GRPO/EasyR1/checkpoints/ot_grpo/qwen2.5_grpo
  /mnt/data/kw/jzs/OT-GRPO/EasyR1/checkpoints/ot_grpo/qwen2.5_grpo
  /mnt/data/kw/jzs/OT-GRPO/EasyR1/checkpoints/ot_grpo/qwen2.5_grpo
  # /mnt/data/kw/jzs/DUCL/train_sft/ckp/qwen-scaling/qwen-scaling_epoch0_step_20940
  # /mnt/data/kw/jzs/DUCL/train_sft/ckp/qwen-scaling/qwen-scaling_epoch0_step_17450
  # /mnt/data/kw/jzs/DUCL/train_sft/ckp/qwen-dsir/qwen-dsir_epoch0_step_20940
  # /mnt/data/kw/jzs/DUCL/train_sft/ckp/qwen-dsir/qwen-dsir_epoch0_step_17450
  # /mnt/data/kw/jzs/DUCL/train_sft/ckp/qwen-dsir/qwen-dsir_epoch0_step_13960
  # /mnt/data/kw/jzs/DUCL/train_sft/ckp/qwen-dsir/qwen-dsir_epoch0_step_10470
  # /mnt/data/kw/jzs/DUCL/train_sft/ckp/qwen-dsir/qwen-dsir_epoch0_step_6980
)

# MODELS=(
#   "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/Qwen2.5-7B_x_OpenR1-7B_spec_grpov3_wo_cliper_w_adv_q85_1/global_step_280/actor/huggingface"
#   /mnt/data/kw/jzs/spec_grpo/mentor/ckp/MENTOR-Qwen-7B
#   # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/Qwen2.5-7B_x_OpenR1-7B_spec_grpov3_wo_cliper_w_adv_q85_1/global_step_280/actor/huggingface"
#   # "/mnt/data/kw/models/Qwen2.5-7B"
#   # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/Qwen2.5-7B_GRPOv2/global_step_195/actor/huggingface"
#   # "/mnt/data/kw/jzs/spec_grpo/LUFFY/luffy/verl/checkpoints/easy_r1/LUFFY_Qwen7b/global_step_200/actor/huggingface"
#   # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/temp/questA-qwen7B/global_step_265/actor/huggingface"
#   # # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/Qwen2.5-7B_x_OpenR1-7B_spec_grpov3_wo_cliper_w_adv_q85_1/global_step_280/actor/huggingface"
#   # "/mnt/data/kw/models/Qwen2.5-3B-resize"
#   # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/Qwen2.5-3B_GRPO/global_step_200/actor/huggingface"
#   # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/Qwen2.5-3B_SpecGRPO/global_step_205/actor/huggingface"
#   # # "/mnt/data/kw/models/Meta-Llama-3.1-8B"

#   # # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/temp/spec-llama3-8b/global_step_140/actor/huggingface"
#   # # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/llama3-8B_GRPO/global_step_225/actor/huggingface"
#   # # # "/mnt/data/kw/jzs/download/model/MENTOR/ckp/qwen3b-questA"
#   # # "/mnt/data/kw/jzs/download/model/MENTOR/ckp/llama3-questA"
#   # # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/temp/spec-llamav2/global_step_200/actor/huggingface"
#   # #  "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/temp/spec-llamav2/global_step_220/actor/huggingface"

# )

# MODELS=(
#   # "/mnt/data/kw/lty/ckp/llama3_8b_jzs/checkpoint-230"
#   # "/mnt/data/kw/lty/ckp/llama3_8b_jzs/checkpoint-276"
#   # "/mnt/data/kw/models/Llama-3.1-8B-R1-Distill"
#   # "/mnt/data/kw/lty/ckp/llama3_8b_jzs/checkpoint-282"

#   # "/mnt/data/kw/models/Qwen2.5-3B-resize"
#   # "/mnt/data/kw/models/Qwen2.5-7B"
#   # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/Qwen2.5-3B_SpecGRPO/global_step_205/actor/huggingface"
#   "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/Qwen2.5-7B_x_OpenR1-7B_spec_grpov3_wo_cliper_w_adv_q85_1/global_step_280/actor/huggingface"
#   # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/Qwen2.5-3B_GRPO/global_step_200/actor/huggingface"
#   # "/mnt/data/kw/jzs/spec_grpo/EasyR1/checkpoints/easy_r1/Qwen2.5-7B_GRPOv2/global_step_195/actor/huggingface"
#   # "Qwen/Qwen2.5-7B"
#   # "Qwen/Qwen2.5-3B"
#   # "/mnt/data/kw/jzs/spec_grpo/LUFFY/luffy/verl/checkpoints/easy_r1/LUFFY_Qwen7b/global_step_200/actor/huggingface"
# )

# TASKS=("math500" "aime24" "aime25" "amc" "minerva" "olympiad" "gpqa_d" 'mmlu_pro' 'arc_c') # "math500" "aime24" "aime25" "amc" "minerva" "olympiad"
TASKS=("math500" "aime24" "aime25") # "math500" "aime24" "aime25" "amc" "minerva" "olympiad"

BATCH_SIZE=999
NUM_WORKERS=0
USE_CHAT_TEMPLATE=1
DTYPE="bfloat16"
ENFORCE_EAGER=0

LOG_DIR="./logs"
OUT_DIR="./results"
mkdir -p "$LOG_DIR" "$OUT_DIR"

for MODEL in "${MODELS[@]}"; do
  SAFE_NAME="${MODEL//\//_}"
  TS="$(date +%Y%m%d-%H%M%S)"
  LOG_FILE="${LOG_DIR}/${TS}-${SAFE_NAME}.log"
  OUT_FILE="${OUT_DIR}/${TS}-${SAFE_NAME}.jsonl"

  echo "=== [$(date '+%F %T')] Start: ${MODEL}"
  echo "    Log: ${LOG_FILE}"
  echo "    Out: ${OUT_FILE}"

  ARGS=( dp_infer.py
    --model_path "${MODEL}"
    --tasks "${TASKS[@]}"
    --batch_size "${BATCH_SIZE}"
    --output "${OUT_FILE}"
    --num_workers "${NUM_WORKERS}"
    --dtype "${DTYPE}"
  )
  [[ "${USE_CHAT_TEMPLATE}" == "1" ]] && ARGS+=( --use_chat_template )
  [[ "${ENFORCE_EAGER}" == "0" ]] && ARGS+=( --no_enforce_eager )

  # 关键：失败不让脚本退出，继续下一个模型
  python -u "${ARGS[@]}" > "${LOG_FILE}" 2>&1 \
    || { echo "!!! ${MODEL} 运行失败，详见 ${LOG_FILE}"; continue; }

  echo "=== [$(date '+%F %T')] Done: ${MODEL}"
  echo
done

echo "All models attempted."

#!/usr/bin/env bash
# MeaCap_InvLM single-image inference (run from project root; paths are relative).
# Usage: bash scripts/run_meacap_invlm.sh [image_path]
# Override models: export LANGUAGE_MODEL=... VL_MODEL=... etc.

set -e
cd "$(dirname "$0")/.."

IMAGE_PATH="${1:-images/instance1.jpg}"
ILR_ARGS=""
if [[ "${USE_ILR:-0}" == "1" ]]; then
  ILR_ARGS="--use_ilr --fusion_w1 0.8 --fusion_w2 0.2 --fusion_type ${FUSION_TYPE:-gated}"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

python viecap_inference.py \
  --device cuda:0 \
  --memory_id coco \
  --image_path "${IMAGE_PATH}" \
  --weight_path checkpoints/train_coco/coco_prefix-0014.pt \
  --using_hard_prompt \
  --soft_prompt_first \
  --language_model "${LANGUAGE_MODEL:-./checkpoints/gpt2}" \
  --vl_model "${VL_MODEL:-./checkpoints/clip-vit-base-patch32}" \
  --parser_checkpoint "${PARSER_CKPT:-./checkpoints/flan-t5-base-VG-factual-sg}" \
  --wte_model_path "${WTE_MODEL:-./checkpoints/all-MiniLM-L6-v2}" \
  --local_files_only \
  ${ILR_ARGS}

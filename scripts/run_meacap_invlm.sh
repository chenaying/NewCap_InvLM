#!/usr/bin/env bash
# MeaCap_InvLM single-image inference (offline-friendly defaults for teacher5 server).
# Usage: bash scripts/run_meacap_invlm.sh [image_path]

set -e
cd "$(dirname "$0")/.."

IMAGE_PATH="${1:-images/instance1.jpg}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

python viecap_inference.py \
  --device cuda:0 \
  --memory_id coco \
  --image_path "${IMAGE_PATH}" \
  --weight_path checkpoints/train_coco/coco_prefix-0014.pt \
  --using_hard_prompt \
  --soft_prompt_first \
  --language_model "${LANGUAGE_MODEL:-/home/teacher5/data1/cyp/project/NewCap/gpt2}" \
  --vl_model "${VL_MODEL:-/home/teacher5/data1/cyp/project/NewCap/checkpoints/clip-vit-base-patch32}" \
  --parser_checkpoint "${PARSER_CKPT:-/home/teacher5/data1/cyp/project/NewCap/checkpoints/flan-t5-base-VG-factual-sg}" \
  --wte_model_path "${WTE_MODEL:-/home/teacher5/data1/cyp/project/NewCap/checkpoints/all-MiniLM-L6-v2}" \
  --local_files_only

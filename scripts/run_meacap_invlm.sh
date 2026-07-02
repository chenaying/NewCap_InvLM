#!/usr/bin/env bash
# MeaCap InvLM single-image inference (run from project root; paths are relative).
#
# Usage:
#   bash scripts/run_meacap_invlm.sh [image_path]
#
# Common overrides (environment variables):
#   USE_ILR=1|0              enable ILR feature fusion (default: 0)
#   FUSION_TYPE=linear|gated   fusion type when USE_ILR=1 (default: gated)
#   FUSION_W1 / FUSION_W2      linear fusion weights (default: 0.8 / 0.2)
#   MEMORY_ID=coco|flickr30k   memory bank for retrieval (default: coco)
#   MEMORY_CAPTION_NUM=5       top-K memory captions (default: 5)
#   WEIGHT_PATH=...            checkpoint path (overrides EXP_NAME + EPOCH)
#   EXP_NAME=train_coco        used with EPOCH if WEIGHT_PATH unset (default: train_coco)
#   EPOCH=14                   checkpoint epoch (default: 14)
#   DEVICE=0                   CUDA device index (default: 0)
#
# Examples:
#   # COCO in-domain, linear
#   USE_ILR=1 FUSION_TYPE=linear bash scripts/run_meacap_invlm.sh images/instance1.jpg
#
#   # Flickr30k in-domain, linear
#   USE_ILR=1 FUSION_TYPE=linear MEMORY_ID=flickr30k \
#     EXP_NAME=train_flickr30k EPOCH=29 \
#     bash scripts/run_meacap_invlm.sh ./annotations/flickr30k/flickr30k-images/1007129816.jpg
#
#   # COCO -> Flickr30k cross-domain, linear
#   USE_ILR=1 FUSION_TYPE=linear MEMORY_ID=flickr30k \
#     EXP_NAME=train_coco EPOCH=14 \
#     bash scripts/run_meacap_invlm.sh ./annotations/flickr30k/flickr30k-images/1007129816.jpg
#
#   # Flickr30k -> COCO cross-domain, linear
#   USE_ILR=1 FUSION_TYPE=linear MEMORY_ID=coco \
#     EXP_NAME=train_flickr30k EPOCH=29 \
#     bash scripts/run_meacap_invlm.sh ./annotations/coco/val2014/COCO_val2014_000000000042.jpg

set -e
cd "$(dirname "$0")/.."

IMAGE_PATH="${1:-images/instance1.jpg}"
DEVICE="${DEVICE:-0}"
MEMORY_ID="${MEMORY_ID:-coco}"
MEMORY_CAPTION_NUM="${MEMORY_CAPTION_NUM:-5}"
EXP_NAME="${EXP_NAME:-train_coco}"
EPOCH="${EPOCH:-14}"
WEIGHT_PATH="${WEIGHT_PATH:-checkpoints/${EXP_NAME}/coco_prefix-$(printf '%04d' "${EPOCH}").pt}"

ILR_ARGS=""
if [[ "${USE_ILR:-0}" == "1" ]]; then
  ILR_ARGS="--use_ilr --fusion_w1 ${FUSION_W1:-0.8} --fusion_w2 ${FUSION_W2:-0.2} --fusion_type ${FUSION_TYPE:-gated}"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

python viecap_inference.py \
  --device "cuda:${DEVICE}" \
  --memory_id "${MEMORY_ID}" \
  --memory_caption_num "${MEMORY_CAPTION_NUM}" \
  --image_path "${IMAGE_PATH}" \
  --weight_path "${WEIGHT_PATH}" \
  --using_hard_prompt \
  --soft_prompt_first \
  --language_model "${LANGUAGE_MODEL:-./checkpoints/gpt2}" \
  --vl_model "${VL_MODEL:-./checkpoints/clip-vit-base-patch32}" \
  --parser_checkpoint "${PARSER_CKPT:-./checkpoints/flan-t5-base-VG-factual-sg}" \
  --wte_model_path "${WTE_MODEL:-./checkpoints/all-MiniLM-L6-v2}" \
  --local_files_only \
  ${ILR_ARGS}

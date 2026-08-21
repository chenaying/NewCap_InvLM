#!/usr/bin/env bash
# MeaCap InvLM batch evaluation on NoCaps (same flow as eval_nocaps.sh).
# Usage (ViECap-compatible, 4 args):
#   bash scripts/eval_nocaps_meacap.sh train_coco 0 '--top_k 3 --threshold 0.2' 14
# Usage (3 args):
#   bash scripts/eval_nocaps_meacap.sh train_coco 0 14

set -e
set -o pipefail
SHELL_FOLDER=$(cd "$(dirname "$0")"; pwd)
cd "${SHELL_FOLDER}/.."

EXP_NAME=${1:-train_coco}
DEVICE=${2:-0}
if [[ $# -ge 4 ]]; then
  EXTRA_ARGS=${3:-}
  EPOCH=${4:-14}
else
  EPOCH=${3:-14}
  EXTRA_ARGS=${4:-}
fi

WEIGHT_PATH="checkpoints/${EXP_NAME}/coco_prefix-$(printf '%04d' "${EPOCH}").pt"
NOCAPS_OUT_PATH="checkpoints/${EXP_NAME}"

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER="logs/${EXP_NAME}_EVAL_MEACAP"
mkdir -p "${LOG_FOLDER}"
NOCAPS_LOG_FILE="${LOG_FOLDER}/NOCAPS_MEACAP_${TIME_START}.log"

LANGUAGE_MODEL="${LANGUAGE_MODEL:-./checkpoints/gpt2}"
VL_MODEL="${VL_MODEL:-./checkpoints/clip-vit-base-patch32}"
PARSER_CKPT="${PARSER_CKPT:-./checkpoints/flan-t5-base-VG-factual-sg}"
WTE_MODEL="${WTE_MODEL:-./checkpoints/all-MiniLM-L6-v2}"
MEMORY_ID="${MEMORY_ID:-coco}"
ILR_ARGS=""
if [[ "${USE_ILR:-0}" == "1" ]]; then
  ILR_ARGS="--use_ilr --fusion_w1 0.8 --fusion_w2 0.2 --fusion_type ${FUSION_TYPE:-gated}"
fi
C3_ARGS=""
if [[ "${REMOVE_MEAN:-0}" == "1" ]]; then
  C3_ARGS="--remove_mean \
    --text_embed_mean_path ${TEXT_EMBED_MEAN_PATH:-./annotations/coco/normalized_text_embed_mean.pt} \
    --image_embed_mean_path ${IMAGE_EMBED_MEAN_PATH:-./annotations/coco/normalized_image_embed_mean.pt}"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

FEATURE_PICKLE="./annotations/nocaps/nocaps_corpus_ViT-B32.pickle"
NOCAPS_IMG_DIR="./annotations/nocaps"
if [[ ! -f "${FEATURE_PICKLE}" ]]; then
  echo "ERROR: missing ${FEATURE_PICKLE}"
  echo "  1) unzip ViECap checkpoints.zip into project root, or"
  echo "  2) python images_features_extraction.py"
  exit 1
fi
echo "Using feature pickle: ${FEATURE_PICKLE} ($(du -h "${FEATURE_PICKLE}" | cut -f1))"

# InvLM needs raw images (HF CLIP); ViECap pickle alone is not enough
if [[ ! -d "${NOCAPS_IMG_DIR}/in_domain" && ! -d "${NOCAPS_IMG_DIR}/in-domain" && ! -d "${NOCAPS_IMG_DIR}/val" ]]; then
  echo "ERROR: NoCaps images not found under ${NOCAPS_IMG_DIR}/"
  echo "  Expected one of: in_domain/, in-domain/, or val/"
  echo "  Download from ViECap checkpoints.zip (annotations/nocaps/) or official NoCaps val images."
  exit 1
fi

python validation.py \
  --device "cuda:${DEVICE}" \
  --clip_model ViT-B/32 \
  --language_model "${LANGUAGE_MODEL}" \
  --continuous_prompt_length 10 \
  --clip_project_length 10 \
  --using_image_features \
  --name_of_datasets nocaps \
  --path_of_val_datasets ./annotations/nocaps/nocaps_corpus.json \
  --image_folder ./annotations/nocaps/ \
  --weight_path "${WEIGHT_PATH}" \
  --out_path "${NOCAPS_OUT_PATH}" \
  --using_hard_prompt \
  --soft_prompt_first \
  --use_meacap_invlm \
  --memory_id "${MEMORY_ID}" \
  --memory_caption_num 5 \
  --vl_model "${VL_MODEL}" \
  --parser_checkpoint "${PARSER_CKPT}" \
  --wte_model_path "${WTE_MODEL}" \
  --local_files_only \
  ${ILR_ARGS} \
  ${C3_ARGS} \
  ${EXTRA_ARGS} \
  2>&1 | tee -a "${NOCAPS_LOG_FILE}"

echo "========================== NOCAPS IN-DOMAIN (MeaCap InvLM) =============================="
python evaluation/cocoeval.py --result_file_path "${NOCAPS_OUT_PATH}/indomain"*.json 2>&1 | tee -a "${NOCAPS_LOG_FILE}"
echo "========================== NOCAPS NEAR-DOMAIN (MeaCap InvLM) ============================="
python evaluation/cocoeval.py --result_file_path "${NOCAPS_OUT_PATH}/neardomain"*.json 2>&1 | tee -a "${NOCAPS_LOG_FILE}"
echo "========================== NOCAPS OUT-DOMAIN (MeaCap InvLM) =============================="
python evaluation/cocoeval.py --result_file_path "${NOCAPS_OUT_PATH}/outdomain"*.json 2>&1 | tee -a "${NOCAPS_LOG_FILE}"
echo "========================== NOCAPS OVERALL (MeaCap InvLM) ================================"
python evaluation/cocoeval.py --result_file_path "${NOCAPS_OUT_PATH}/overall"*.json 2>&1 | tee -a "${NOCAPS_LOG_FILE}"

#!/usr/bin/env bash
# Sequential COCO + Flickr30k gated pipeline (data prep → train → eval).
# Usage:
#   nohup bash scripts/run_all_gated_pipeline.sh 0 > run_all_gated_total.log 2>&1 &
#   tail -f run_all_gated_total.log

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${1:-0}"
export USE_ILR=1
export FUSION_TYPE="${FUSION_TYPE:-gated}"
export FUSION_TEMPERATURE="${FUSION_TEMPERATURE:-0.07}"

COCO_PICKLE="./annotations/coco/coco_texts_features_ViT-B32.pickle"
COCO_ILR="./annotations/coco/coco_ilr_neighbors_k5_seed30_var0.04.json"
FLICKR_PICKLE="./annotations/flickr30k/flickr30k_texts_features_ViT-B32.pickle"
FLICKR_ILR="./annotations/flickr30k/flickr30k_ilr_neighbors_k5_seed30_var0.04.json"

log() { echo "[$(date '+%F %T')] $*"; }

prepare_flickr30k_data() {
  log "===== Flickr30k data prep (entities + CLIP features + ILR) ====="
  python - <<'PY'
import os
import clip
import pickle
import torch
from nltk.stem import WordNetLemmatizer
import nltk
from load_annotations import load_captions

def extract_entities(captions, out_path):
    if os.path.exists(out_path):
        print(f"Skip entities (exists): {out_path}")
        return
    lemmatizer = WordNetLemmatizer()
    rows = []
    for caption in captions:
        ents = []
        for word, pos in nltk.pos_tag(nltk.word_tokenize(caption)):
            if pos in ("NN", "NNS"):
                ents.append(lemmatizer.lemmatize(word.lower().strip()))
        rows.append([list(set(ents)), caption])
    with open(out_path, "wb") as f:
        pickle.dump(rows, f)
    print(f"Wrote entities: {out_path} ({len(rows)} samples)")

def extract_clip_features(in_path, out_path, device="cuda:0"):
    if os.path.exists(out_path):
        print(f"Skip CLIP features (exists): {out_path}")
        return
    encoder, _ = clip.load("ViT-B/32", device=device)
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    for i in range(len(data)):
        tokens = clip.tokenize(data[i][1], truncate=True).to(device)
        data[i].append(encoder.encode_text(tokens).squeeze(0).cpu())
        if (i + 1) % 50000 == 0:
            print(f"  CLIP encoded {i+1}/{len(data)}")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote CLIP features: {out_path}")

in_entities = "./annotations/flickr30k/flickr30k_with_entities.pickle"
out_pickle = "./annotations/flickr30k/flickr30k_texts_features_ViT-B32.pickle"
captions = load_captions("flickr30k_captions", "./annotations/flickr30k/train_captions.json")
extract_entities(captions, in_entities)
extract_clip_features(in_entities, out_pickle)
PY
  python ilr/build_ilr_neighbors.py \
    --path_of_datasets "${FLICKR_PICKLE}" \
    --output_path "${FLICKR_ILR}"
}

log "===== Step 1: COCO entities_extraction.py ====="
python entities_extraction.py

log "===== Step 2: COCO texts_features_extraction.py ====="
python texts_features_extraction.py

log "===== Step 3: COCO build_ilr_neighbors ====="
python ilr/build_ilr_neighbors.py \
  --path_of_datasets "${COCO_PICKLE}" \
  --output_path "${COCO_ILR}"

log "===== Step 4: train_coco gated ====="
bash scripts/train_coco.sh "${DEVICE}"

log "===== Step 5: eval_coco_meacap epoch14 ====="
bash scripts/eval_coco_meacap.sh train_coco "${DEVICE}" 14

log "===== Step 6: eval_flickr30k MEMORY_ID=coco ====="
MEMORY_ID=coco bash scripts/eval_flickr30k_meacap.sh train_coco "${DEVICE}" 14 \
  > eval_flickr_mem_coco.log 2>&1

log "===== Step 7: eval_flickr30k MEMORY_ID=flickr30k ====="
MEMORY_ID=flickr30k bash scripts/eval_flickr30k_meacap.sh train_coco "${DEVICE}" 14 \
  > eval_flickr_mem_flickr.log 2>&1

prepare_flickr30k_data

log "===== Step 8: train_flickr30k gated ====="
bash scripts/train_flickr30k.sh "${DEVICE}"

log "===== Step 9: eval_flickr30k_meacap epoch29 ====="
bash scripts/eval_flickr30k_meacap.sh train_flickr30k "${DEVICE}" 29 \
  > eval_flickr30k.log 2>&1

log "===== Step 10: eval_coco_meacap MEMORY_ID=coco (flickr weights) ====="
MEMORY_ID=coco bash scripts/eval_coco_meacap.sh train_flickr30k "${DEVICE}" 29 \
  > eval_coco_mem_coco.log 2>&1

log "===== Step 11: eval_coco_meacap MEMORY_ID=flickr30k (flickr weights) ====="
MEMORY_ID=flickr30k bash scripts/eval_coco_meacap.sh train_flickr30k "${DEVICE}" 29 \
  > eval_coco_mem_flickr.log 2>&1

log "######## ALL gated pipeline completed ########"

#!/usr/bin/env bash
# Offline ILR neighbor precomputation for Flickr30k training set.
set -e
SHELL_FOLDER=$(cd "$(dirname "$0")"; pwd)
cd "${SHELL_FOLDER}/.."

DEVICE=${1:-0}

python ilr/build_ilr_neighbors.py \
  --path_of_datasets ./annotations/flickr30k/flickr30k_texts_features_ViT-B32.pickle \
  --output_path ./annotations/flickr30k/flickr30k_ilr_neighbors_k5_seed30_var0.04.json \
  --ilr_k 5 \
  --ilr_variance 0.04 \
  --seed 30 \
  --device "cuda:${DEVICE}"

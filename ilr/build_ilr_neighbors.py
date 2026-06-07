"""
Offline precompute ILR neighbor indices for training.

Usage:
  python ilr/build_ilr_neighbors.py \
    --path_of_datasets ./annotations/coco/coco_texts_features_ViT-B32.pickle \
    --output_path ./annotations/coco/coco_ilr_neighbors_k5_seed30_var0.04.json
"""

import argparse
import json
import os
import pickle
import sys

import torch

# allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ilr.cosine_retrieval import build_neighbor_index_list


def main():
    parser = argparse.ArgumentParser(description='Build offline ILR neighbor indices (cosine similarity).')
    parser.add_argument(
        '--path_of_datasets',
        default='./annotations/coco/coco_texts_features_ViT-B32.pickle',
        help='pickle with [[entities, caption, clip_feature], ...]',
    )
    parser.add_argument('--output_path', default='./annotations/coco/coco_ilr_neighbors_k5_seed30_var0.04.json')
    parser.add_argument('--ilr_k', type=int, default=5, help='number of retrieved neighbors per caption')
    parser.add_argument('--ilr_variance', type=float, default=0.04, help='noise variance for ILR query simulation')
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    with open(args.path_of_datasets, 'rb') as f:
        data = pickle.load(f)

    caption_features = torch.stack([item[2] for item in data])
    print(f'Loaded {caption_features.shape[0]} captions, dim={caption_features.shape[1]}')

    neighbors = build_neighbor_index_list(
        caption_features,
        k=args.ilr_k,
        variance=args.ilr_variance,
        device=args.device,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    payload = {
        'k': args.ilr_k,
        'variance': args.ilr_variance,
        'seed': args.seed,
        'source': args.path_of_datasets,
        'num_samples': len(neighbors),
        'neighbors': neighbors,
    }
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)

    print(f'Saved ILR neighbors to {args.output_path}')


if __name__ == '__main__':
    main()

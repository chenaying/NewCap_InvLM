"""Cosine-similarity ILR (Image-Like Retrieval) for offline neighbor precomputation."""

import random
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from utils import noise_injection


def cosine_topk_indices(
    query: torch.Tensor,
    corpus: torch.Tensor,
    k: int,
    exclude_idx: Optional[int] = None,
) -> List[int]:
    """
    Args:
        query: (clip_dim,) L2-normalized query feature
        corpus: (N, clip_dim) L2-normalized corpus features
        k: number of neighbors
        exclude_idx: index to exclude (typically self)
    Return:
        list of k corpus indices
    """
    similarity = query @ corpus.T
    if exclude_idx is not None:
        similarity = similarity.clone()
        similarity[exclude_idx] = 0
    neighbors = []
    sim = similarity.clone()
    for _ in range(k):
        max_id = int(torch.max(sim, dim=0).indices.item())
        neighbors.append(max_id)
        sim[max_id] = 0
    return neighbors


def build_neighbor_index_list(
    caption_features: torch.Tensor,
    k: int,
    variance: float,
    device: str = 'cuda:0',
    seed: int = 30,
) -> List[List[int]]:
    """
    Offline ILR: noise-inject each caption feature, retrieve top-K via cosine similarity.

    Args:
        caption_features: (N, clip_dim)
        k: neighbors per caption
        variance: noise variance for image-like simulation (IFCap default 0.04)
        device: device for noise injection
        seed: random seed
    Return:
        neighbors[i] = list of k indices into caption_features
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    corpus = caption_features.float().to(device)
    corpus = torch.nn.functional.normalize(corpus, dim=-1)

    noise_features = noise_injection(corpus, variance=variance, device=device).float()

    neighbors_list = []
    for i in tqdm(range(corpus.shape[0]), desc='ILR cosine retrieval'):
        query = noise_features[i]
        neighbors_list.append(cosine_topk_indices(query, corpus, k, exclude_idx=i))

    return neighbors_list

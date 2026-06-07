"""Shared MeaCap InvLM prompt building for viecap_inference.py and validation.py."""

import copy
import os
import json
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from models.clip_utils import CLIP
from meacap_utils.detect_utils import retrieve_concepts
from utils import compose_discrete_prompts, fuse_clip_features

cpu_device = torch.device('cpu')


def hf_load_kwargs(local_files_only: bool) -> dict:
    return {'local_files_only': True} if local_files_only else {}


def load_memory_clip(model_path: str, local_files_only: bool) -> CLIP:
    import inspect

    if local_files_only:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

    params = inspect.signature(CLIP.__init__).parameters
    if 'local_files_only' in params:
        return CLIP(model_path, local_files_only=local_files_only)
    return CLIP(model_path)


class MeaCapInvLMResources:
    """Memory bank + retrieval models (load once for batch evaluation)."""

    def __init__(self, args, device: torch.device):
        hf_kw = hf_load_kwargs(getattr(args, 'local_files_only', False))

        self.device = device
        self.memory_caption_num = args.memory_caption_num

        self.vl_model = load_memory_clip(args.vl_model, getattr(args, 'local_files_only', False))
        self.vl_model = self.vl_model.to(device)

        self.wte_model = SentenceTransformer(args.wte_model_path)
        self.parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint, **hf_kw)
        self.parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint, **hf_kw)
        self.parser_model.eval()
        self.parser_model.to(device)

        memory_id = args.memory_id
        memory_dir = os.path.join('data/memory', memory_id)
        with open(os.path.join(memory_dir, 'memory_captions.json'), 'r', encoding='utf-8') as f:
            self.memory_captions = json.load(f)
        self.memory_clip_embeddings = torch.load(
            os.path.join(memory_dir, 'memory_clip_embeddings.pt'),
            map_location=device,
        )

        if memory_id in ('cc3m', 'ss1m'):
            self.retrieve_on_cpu = True
            self.vl_model_retrieve = copy.deepcopy(self.vl_model).to(cpu_device)
            self.memory_clip_embeddings = self.memory_clip_embeddings.to(cpu_device)
        else:
            self.retrieve_on_cpu = False
            self.vl_model_retrieve = self.vl_model

        print(f'[MeaCap InvLM] memory bank "{memory_id}" loaded ({len(self.memory_captions)} captions).')


def _memory_cosine_topk(resources: MeaCapInvLMResources, batch_image_embeds: torch.Tensor):
    device = resources.device
    if not resources.retrieve_on_cpu:
        clip_score, raw_logits = resources.vl_model_retrieve.compute_image_text_similarity_via_embeddings(
            batch_image_embeds, resources.memory_clip_embeddings
        )
    else:
        batch_cpu = batch_image_embeds.to(cpu_device)
        clip_score, raw_logits = resources.vl_model_retrieve.compute_image_text_similarity_via_embeddings(
            batch_cpu, resources.memory_clip_embeddings
        )
        clip_score = clip_score.to(device)
        raw_logits = raw_logits.to(device)

    k = resources.memory_caption_num
    select_ids = clip_score.topk(k, dim=-1)[1].squeeze(0)
    rt_features = resources.memory_clip_embeddings[select_ids]
    if resources.retrieve_on_cpu:
        rt_features = rt_features.to(device)
    return select_ids, rt_features, raw_logits


def retrieve_memory_rt_features(resources: MeaCapInvLMResources, image_path: str) -> torch.Tensor:
    """Cosine top-K memory caption CLIP embeddings, shape (K, clip_dim)."""
    batch_image_embeds = resources.vl_model.compute_image_representation_from_image_path(image_path)
    _, rt_features, _ = _memory_cosine_topk(resources, batch_image_embeds)
    return rt_features.float()


def compute_continuous_embeddings(
    args,
    model,
    image_features: torch.Tensor,
    invlm_resources: MeaCapInvLMResources = None,
    image_path: str = None,
) -> torch.Tensor:
    """
    MappingNetwork input with optional ILR fusion:
    e_fused = w1 * e_img + w2 * mean(retrieved_memory_features)
    """
    primary = image_features
    if getattr(args, 'use_ilr', False):
        if invlm_resources is None or image_path is None:
            raise ValueError('ILR fusion at inference requires MeaCap memory bank and image_path.')
        rt_features = retrieve_memory_rt_features(invlm_resources, image_path)
        primary = fuse_clip_features(
            primary,
            rt_features.unsqueeze(0),
            getattr(args, 'fusion_w1', 0.85),
            getattr(args, 'fusion_w2', 0.15),
        )
    return model.mapping_network(primary).view(
        -1, args.continuous_prompt_length, model.gpt_hidden_size
    )


def retrieve_memory_concepts(resources: MeaCapInvLMResources, image_path: str) -> List[str]:
    device = resources.device
    batch_image_embeds = resources.vl_model.compute_image_representation_from_image_path(image_path)

    select_ids, _, _ = _memory_cosine_topk(resources, batch_image_embeds)
    select_captions = [resources.memory_captions[i] for i in select_ids]

    return retrieve_concepts(
        parser_model=resources.parser_model,
        parser_tokenizer=resources.parser_tokenizer,
        wte_model=resources.wte_model,
        select_memory_captions=select_captions,
        image_embeds=batch_image_embeds,
        device=device,
    )


def build_embeddings(
    args,
    model,
    tokenizer,
    continuous_embeddings: torch.Tensor,
    detected_objects: List[str],
    device: torch.device,
) -> torch.Tensor:
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim=0).to(device)
    discrete_embeddings = model.word_embed(discrete_tokens)

    if args.only_hard_prompt:
        return discrete_embeddings
    if args.soft_prompt_first:
        return torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
    return torch.cat((discrete_embeddings, continuous_embeddings), dim=1)


NOCAPS_SPLIT_DIR_ALIASES = {
    'in_domain': ('in_domain', 'in-domain', 'In-Domain', 'indomain'),
    'near_domain': ('near_domain', 'near-domain', 'Near-Domain', 'neardomain'),
    'out_domain': ('out_domain', 'out-domain', 'Out-Domain', 'outdomain'),
}


def _normalize_image_id(image_id) -> str:
    image_id = str(image_id).strip()
    if not os.path.splitext(image_id)[1]:
        image_id = f'{image_id}.jpg'
    return image_id


def resolve_image_path(args, image_id: str, split: str = None) -> str:
    """Build filesystem path for HF CLIP memory retrieval (must exist on disk)."""
    image_id = _normalize_image_id(image_id)
    image_folder = args.image_folder.rstrip('/\\')

    candidates = []
    if split is not None:
        split_dirs = NOCAPS_SPLIT_DIR_ALIASES.get(split, (split,))
        for split_dir in split_dirs:
            candidates.append(os.path.join(image_folder, split_dir, image_id))
        # official NoCaps val layout: annotations/nocaps/val/<file>
        candidates.append(os.path.join(image_folder, 'val', image_id))
    else:
        candidates.append(os.path.join(image_folder, image_id))

    for path in candidates:
        if os.path.isfile(path):
            return path

    # basename index built once for NoCaps (optional fallback)
    index = getattr(args, '_nocaps_image_index', None)
    if index is not None:
        if image_id in index:
            return index[image_id]
        stem = os.path.splitext(image_id)[0]
        if stem in index:
            return index[stem]

    tried = '\n  '.join(candidates[:6])
    raise FileNotFoundError(
        f'NoCaps image not found for image_id={image_id}, split={split}.\n'
        f'Tried:\n  {tried}\n'
        f'Please download NoCaps val images into {image_folder}/in_domain/ (and near/out_domain).\n'
        f'ViECap checkpoints.zip usually includes annotations/nocaps/ with images, or run images_features_extraction.py after placing images.'
    )


def build_nocaps_image_index(image_folder: str) -> dict:
    """Map basename / stem -> absolute image path under image_folder."""
    index = {}
    if not os.path.isdir(image_folder):
        return index
    for dirpath, _, filenames in os.walk(image_folder):
        for fname in filenames:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            full = os.path.join(dirpath, fname)
            index[fname] = full
            stem, _ = os.path.splitext(fname)
            index.setdefault(stem, full)
    return index


def check_nocaps_images_available(args, sample_paths: List[tuple]) -> None:
    """Preflight: verify a few NoCaps images exist before 4500-iter loop."""
    if not getattr(args, 'use_meacap_invlm', False):
        return
    if not getattr(args, '_nocaps_image_index', None):
        args._nocaps_image_index = build_nocaps_image_index(args.image_folder)
    missing = 0
    for image_id, split in sample_paths[:5]:
        try:
            resolve_image_path(args, image_id, split)
        except FileNotFoundError:
            missing += 1
    if missing == len(sample_paths[:5]):
        raise FileNotFoundError(
            f'No NoCaps images found under {args.image_folder}. '
            'InvLM requires raw images for HF CLIP memory retrieval. '
            'ViECap baseline (--use_meacap_invlm False) can run with pickle features only.'
        )


def hard_prompt_embeddings(
    args,
    model,
    tokenizer,
    continuous_embeddings: torch.Tensor,
    image_features: torch.Tensor,
    image_path: str,
    device: torch.device,
    invlm_resources: MeaCapInvLMResources = None,
    entities_text: List[str] = None,
    texts_embeddings: torch.Tensor = None,
) -> torch.Tensor:
    """ViECap entity classifier or MeaCap memory concepts."""
    if getattr(args, 'use_meacap_invlm', False):
        detected_objects = retrieve_memory_concepts(invlm_resources, image_path)
    else:
        from retrieval_categories import image_text_simiarlity, top_k_categories

        logits = image_text_simiarlity(
            texts_embeddings, temperature=args.temperature, images_features=image_features
        )
        detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
        detected_objects = detected_objects[0]

    return build_embeddings(args, model, tokenizer, continuous_embeddings, detected_objects, device)

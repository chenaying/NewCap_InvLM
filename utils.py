import math
import torch
import torch.nn as nn
import random
import torch.nn.functional as nnf
from typing import List, Optional, Tuple, Union
        
def fuse_clip_features(
    e_primary: torch.Tensor,
    e_retrieved: torch.Tensor,
    w1: float = 0.8,
    w2: float = 0.2,
) -> torch.Tensor:
    """
    Fuse primary CLIP feature with retrieved neighbor features.

    Args:
        e_primary: (B, clip_dim) main feature (noisy text feat in training / image feat at inference)
        e_retrieved: (B, K, clip_dim) or (B, clip_dim) aggregated neighbors
    Return:
        (B, clip_dim) L2-normalized fused feature
    """
    if e_retrieved.dim() == 2:
        e_agg = e_retrieved
    else:
        e_agg = e_retrieved.mean(dim=1)
    e_fused = w1 * e_primary + w2 * e_agg
    return nnf.normalize(e_fused, dim=-1)


class GatedFusion(nn.Module):
    """External gated fusion in CLIP space (before Projector).

    Replaces the fixed-weight linear fusion with a learnable, sample-adaptive,
    per-dimension gate:
        e_agg   = mean(e_retrieved)
        g       = sigmoid(MLP([e_primary; e_agg]))      # (B, clip_dim), per-dim
        e_fused = (1 - g) * e_primary + g * e_agg
    The gate bias is initialized so that the initial gate is ~init_gate (default
    0.2), matching the previously validated linear weight for a stable start.
    """

    def __init__(self, dim: int = 512, init_gate: float = 0.2) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        # logit(init_gate) so that sigmoid(bias) == init_gate at init
        init_gate = min(max(init_gate, 1e-4), 1 - 1e-4)
        bias_value = math.log(init_gate / (1 - init_gate))
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, bias_value)

    def forward(self, e_primary: torch.Tensor, e_retrieved: torch.Tensor) -> torch.Tensor:
        if e_retrieved.dim() == 2:
            e_agg = e_retrieved
        else:
            e_agg = e_retrieved.mean(dim=1)
        g = self.gate(torch.cat([e_primary, e_agg], dim=-1))
        e_fused = (1 - g) * e_primary + g * e_agg
        return nnf.normalize(e_fused, dim=-1)


class GatedCrossAttnFusion(nn.Module):
    """External gated cross-attention fusion in CLIP space (before Projector).

    Cross-attention replaces mean pooling over K retrieved neighbors; the gate
    then blends primary and attended retrieval features per dimension:
        e_attn  = CrossAttn(Q=e_primary, K/V=e_retrieved)
        g       = sigmoid(MLP([e_primary; e_attn]))
        e_fused = (1 - g) * e_primary + g * e_attn
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, init_gate: float = 0.2) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        init_gate = min(max(init_gate, 1e-4), 1 - 1e-4)
        bias_value = math.log(init_gate / (1 - init_gate))
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, bias_value)

    def forward(self, e_primary: torch.Tensor, e_retrieved: torch.Tensor) -> torch.Tensor:
        if e_retrieved.dim() == 2:
            e_retrieved = e_retrieved.unsqueeze(1)
        q = e_primary.unsqueeze(1)
        e_attn, _ = self.cross_attn(q, e_retrieved, e_retrieved)
        e_attn = e_attn.squeeze(1)
        g = self.gate(torch.cat([e_primary, e_attn], dim=-1))
        e_fused = (1 - g) * e_primary + g * e_attn
        return nnf.normalize(e_fused, dim=-1)


class CrossAttnFusion(nn.Module):
    """External cross-attention fusion in CLIP space (before Projector).

    No gating: the fused feature is the attention-weighted retrieval vector only:
        e_fused = normalize(CrossAttn(Q=e_primary, K/V=e_retrieved))
    """

    def __init__(self, dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, e_primary: torch.Tensor, e_retrieved: torch.Tensor) -> torch.Tensor:
        if e_retrieved.dim() == 2:
            e_retrieved = e_retrieved.unsqueeze(1)
        q = e_primary.unsqueeze(1)
        e_attn, _ = self.cross_attn(q, e_retrieved, e_retrieved)
        return nnf.normalize(e_attn.squeeze(1), dim=-1)


class InternalGatedFusion(nn.Module):
    """Internal gated fusion in LM hidden space (inside MappingNetwork).

    Applied after projecting primary/retrieved CLIP features to d_model tokens:
        e_agg   = mean(rtf, dim=1), broadcast to each query token
        g       = sigmoid(MLP([q_token; e_agg]))   per token, per-dim
        q_out   = (1 - g) * q_token + g * e_agg
    """

    def __init__(self, dim: int = 768, init_gate: float = 0.2) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        init_gate = min(max(init_gate, 1e-4), 1 - 1e-4)
        bias_value = math.log(init_gate / (1 - init_gate))
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, bias_value)

    def forward(self, q_tokens: torch.Tensor, rtf: torch.Tensor) -> torch.Tensor:
        if rtf.dim() == 2:
            rtf = rtf.unsqueeze(1)
        e_agg = rtf.mean(dim=1, keepdim=True).expand_as(q_tokens)
        g = self.gate(torch.cat([q_tokens, e_agg], dim=-1))
        return (1 - g) * q_tokens + g * e_agg


class InternalGatedCrossAttnFusion(nn.Module):
    """Internal gated cross-attention fusion in LM hidden space (inside MappingNetwork).

    Applied after projecting primary/retrieved CLIP features to d_model tokens:
        e_attn  = CrossAttn(Q=q_tokens, K/V=rtf)
        g       = sigmoid(MLP([q_token; e_attn_l]))   per token, per-dim
        q_out   = (1 - g) * q_token + g * e_attn
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, init_gate: float = 0.2) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        init_gate = min(max(init_gate, 1e-4), 1 - 1e-4)
        bias_value = math.log(init_gate / (1 - init_gate))
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, bias_value)

    def forward(self, q_tokens: torch.Tensor, rtf: torch.Tensor) -> torch.Tensor:
        if rtf.dim() == 2:
            rtf = rtf.unsqueeze(1)
        e_attn, _ = self.cross_attn(q_tokens, rtf, rtf)
        g = self.gate(torch.cat([q_tokens, e_attn], dim=-1))
        return (1 - g) * q_tokens + g * e_attn


class InternalResGatedCrossAttnFusion(nn.Module):
    """Internal residual gated cross-attention fusion (IFCap-style residual + gate).

    IFCap uses Q' = Q + CrossAttn(Q, R). Here the retrieval increment is gated:
        delta  = CrossAttn(Q=q_tokens, K/V=rtf)
        g      = sigmoid(MLP([q_token; delta_l]))   per token, per-dim
        q_out  = q_token + g * delta                 primary always preserved
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, init_gate: float = 0.2) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        init_gate = min(max(init_gate, 1e-4), 1 - 1e-4)
        bias_value = math.log(init_gate / (1 - init_gate))
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, bias_value)

    def forward(self, q_tokens: torch.Tensor, rtf: torch.Tensor) -> torch.Tensor:
        if rtf.dim() == 2:
            rtf = rtf.unsqueeze(1)
        delta, _ = self.cross_attn(q_tokens, rtf, rtf)
        g = self.gate(torch.cat([q_tokens, delta], dim=-1))
        return q_tokens + g * delta


def uses_internal_fusion(fusion_type: str) -> bool:
    return fusion_type in (
        'internal_gated',
        'internal_gated_crossattn',
        'internal_resgated_crossattn',
        'internal_ifcap',
    )


def uses_internal_gated(fusion_type: str) -> bool:
    return uses_internal_fusion(fusion_type)


def build_internal_fusion_module(
    fusion_type: str,
    dim: int = 768,
    num_heads: int = 8,
) -> Optional[nn.Module]:
    if fusion_type == 'internal_gated':
        return InternalGatedFusion(dim)
    if fusion_type == 'internal_gated_crossattn':
        return InternalGatedCrossAttnFusion(dim, num_heads)
    if fusion_type == 'internal_resgated_crossattn':
        return InternalResGatedCrossAttnFusion(dim, num_heads)
    if fusion_type == 'internal_ifcap':
        from ifcap_fusion import InternalIFCapFusion
        return InternalIFCapFusion(dim, num_heads, num_layers=1)
    return None


def build_fusion_module(fusion_type: str, dim: int = 512) -> Optional[nn.Module]:
    if fusion_type == 'gated':
        return GatedFusion(dim)
    if fusion_type == 'crossattn':
        return CrossAttnFusion(dim)
    if fusion_type == 'gated_crossattn':
        return GatedCrossAttnFusion(dim)
    return None


def noise_injection(x, variance = 0.001, device = 'cuda:0') -> torch.Tensor:
    """
    Args:
        x: tensor with a shape of (batch_size, clip_hidden_size), prefix
        variance: the variance of noise
    Return:
        prefix with noise
    """
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    # normalization
    x = torch.nn.functional.normalize(x, dim = -1)
    # adding noise
    x = x + (torch.randn(x.shape, device = device) * std)

    return torch.nn.functional.normalize(x, dim = -1)

def entities_process(
    args,
    detected_entities: List[str],  # [man, dog, park]
    stopwords: List[str],
    people_vocabs: List[str],
    objects_vocabs: List[str],
) -> List[str]:
    process_entities = []
    for i in range(len(detected_entities)):
        if i >= args.max_num_of_entities: # There is no entity detected
            break
        detected_entity = detected_entities[i]                # processing the i-th entity
        if detected_entity in people_vocabs:                  # transforming all person concept (man, woman, kid, ...) to the same word 'person'
            detected_entity = 'person'
        elif len(detected_entity) > 1 and detected_entity not in stopwords and detected_entity in objects_vocabs: # only processing entities in visual genome
            pass
        else: # processing the next entities
            continue

        if args.random_mask:
            random_prob = random.random()
            if random_prob < args.prob_of_random_mask:         # mask
                pass
            else:                                              # remain
                process_entities.append(detected_entity)

        else: # entities with any process
            return detected_entities
    
    return process_entities

def compose_discrete_prompts(
    tokenizer,
    process_entities: List[str],
) -> torch.Tensor:

    prompt_head = 'There are'
    prompt_tail = ' in image.'

    if len(process_entities) == 0: # without entities
        discrete_prompt =  prompt_head + ' something' + prompt_tail
    else:
        discrete_prompt = ''
        for entity in process_entities: # gpt2 in transformer encoder ' ' + word into one token by default
            discrete_prompt += ' ' + entity + ','     # ' person, dog, park,'
        discrete_prompt = discrete_prompt[:-1]        # ' person, dog, park'
        discrete_prompt = prompt_head + discrete_prompt + prompt_tail # 'There are person, dog, park in image.'

    entities_tokens = torch.tensor(tokenizer.encode(discrete_prompt))   # (discrete_prompt_length, ) 

    return entities_tokens

def parse_entities(
    args,
    tokenizer,
    detected_entities: Tuple[str],      # [[man, dog, park, ...], len = batch size
    stopwords: List[str],
    people_vocabs: List[str],
    objects_vocabs: List[str],
) -> List[torch.Tensor]:
    # List[(n_seq1, ), (n_seq2, ), ...]

    discrete_tokens = []
    for idx in range(len(detected_entities)):
        # entities processing
        process_entities = entities_process(args, detected_entities[idx], stopwords, people_vocabs, objects_vocabs)
        process_entities = list(set(process_entities)) # list

        # tokenizing
        discrete_tokens.append(compose_discrete_prompts(tokenizer, process_entities))

    return discrete_tokens

def padding_captions(
    args,
    captions_tokens: torch.Tensor,   # (batch_size, caption_seq)
    masks: torch.Tensor,             # (batch_size, caption_seq)
    discrete_tokens: List[torch.Tensor] = None, # len = batch_size, [(n_seq1, ), (n_seq2, ), ...]
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, List]:
    """
    Return:
        captions_tokens:
        captions_tokens_for_loss:
        masks:
        hard_prompts_length: 
    """
    if discrete_tokens is None: # capdec
        masks = torch.cat((torch.ones(len(masks), args.continuous_prompt_length), masks), dim = -1) # (batch_size, continuous_prompt_length + caption_seq)
        captions_tokens_for_loss = torch.cat((torch.zeros((len(captions_tokens), args.continuous_prompt_length), dtype = torch.int64), captions_tokens), dim = -1) # (batch_size, continuous_prompt_length + caption_seq)
        captions_tokens_for_loss = torch.cat((captions_tokens_for_loss[:, 1:], torch.zeros((len(captions_tokens), 1), dtype = torch.int64)), dim = -1)
        return captions_tokens, captions_tokens_for_loss, masks

    else: # discrete tokens
        captions_tokens_with_hard_prompts = None
        captions_tokens_for_loss = None
        padding_masks = None
        hard_prompts_length = []
        max_length = 2 * args.max_num_of_entities - 1 + args.prompt_template_length + captions_tokens.shape[-1] # max length without soft prompt
        for i in range(len(discrete_tokens)):
            tokens = torch.cat((discrete_tokens[i], captions_tokens[i]))
            loss_tokens = torch.cat((torch.zeros((len(discrete_tokens[i])), dtype = torch.int64), captions_tokens[i]))
            padding = max_length - len(tokens)
            if padding > 0:
                tokens = torch.cat((tokens, torch.zeros((padding), dtype = torch.int64)))
                loss_tokens = torch.cat((loss_tokens, torch.zeros((padding), dtype = torch.int64)))
            tokens = tokens[:max_length].unsqueeze(dim = 0) # (1, max_length)
            if args.only_hard_prompt:
                loss_tokens = loss_tokens[:max_length]
            else:
                loss_tokens = torch.cat((torch.zeros((args.continuous_prompt_length), dtype = torch.int64), loss_tokens[:max_length]))
            loss_tokens = torch.cat((loss_tokens[1:], torch.zeros((1), dtype = torch.int64))).unsqueeze(dim = 0) # (1, max_length + continuous_prompt_length)
            if captions_tokens_with_hard_prompts is None:
                captions_tokens_with_hard_prompts = tokens
                captions_tokens_for_loss = loss_tokens
            else:
                captions_tokens_with_hard_prompts = torch.cat((captions_tokens_with_hard_prompts, tokens), dim = 0)
                captions_tokens_for_loss = torch.cat((captions_tokens_for_loss, loss_tokens), dim = 0)

            hard_prompts_length.append(len(discrete_tokens[i]))
            if padding > 0:
                if args.only_hard_prompt:
                    temp_masks = torch.cat((torch.ones(hard_prompts_length[-1]).float(), masks[i], torch.zeros(padding).float()))
                else:
                    temp_masks = torch.cat((torch.ones(hard_prompts_length[-1] + args.continuous_prompt_length).float(), masks[i], torch.zeros(padding).float()))
            else:
                if args.only_hard_prompt:
                    temp_masks = torch.cat((torch.ones(hard_prompts_length[-1]).float(), masks[i]))
                else:
                    temp_masks = torch.cat((torch.ones(hard_prompts_length[-1] + args.continuous_prompt_length).float(), masks[i]))
            if args.only_hard_prompt:
                temp_masks = temp_masks[:max_length].unsqueeze(dim = 0)
            else:
                temp_masks = temp_masks[:max_length + args.continuous_prompt_length].unsqueeze(dim = 0)
            if padding_masks is None:
                padding_masks = temp_masks
            else:
                padding_masks = torch.cat((padding_masks, temp_masks), dim = 0)
        return captions_tokens_with_hard_prompts, captions_tokens_for_loss, padding_masks, hard_prompts_length
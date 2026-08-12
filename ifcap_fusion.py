"""IFCap-style internal cross-attention fusion (att_gt_n_rt) for MeaCapInvLM."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as nnf


class _IFCapMlp(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        act=nnf.relu,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        output_size = output_size if output_size is not None else input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = act
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class _IFCapMultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_size: int,
        key_value_size: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = query_size // num_heads
        self.scale = self.head_size ** -0.5
        self.to_queries = nn.Linear(query_size, query_size, bias=bias)
        self.to_keys_values = nn.Linear(key_value_size, 2 * query_size, bias=bias)
        self.project = nn.Linear(query_size, query_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_value = key_value if key_value is not None else query
        b, n, d_query = query.shape
        _, m, _ = key_value.shape
        queries = self.to_queries(query).reshape(b, n, self.num_heads, self.head_size)
        keys_values = self.to_keys_values(key_value).reshape(b, m, 2, self.num_heads, self.head_size)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(dim=1)
            attention = attention.masked_fill(mask.unsqueeze(dim=3), float('-inf'))

        attention = attention.softmax(dim=2)
        outputs = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, d_query)
        outputs = self.project(outputs)
        return outputs, attention


class _IFCapTransformerLayer(nn.Module):
    def __init__(
        self,
        query_size: int,
        key_value_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        dropout: float = 0.0,
        act=nnf.relu,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(query_size)
        self.attn = _IFCapMultiHeadAttention(query_size, key_value_size, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(query_size)
        self.mlp = _IFCapMlp(query_size, int(query_size * mlp_ratio), act=act, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_, self.attention = self.attn(self.norm1(query), key_value, mask)
        query = query + query_
        query = query + self.mlp(self.norm2(query))
        return query


class _IFCapCrossAttnStack(nn.Module):
    """IFCap att_gt_n_rt: cross-attention stack with fixed K/V across layers."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _IFCapTransformerLayer(
                d_model,
                d_model,
                num_heads,
                mlp_ratio=mlp_ratio,
                bias=False,
            )
            for _ in range(num_layers)
        ])

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, key_value)
        return query


class InternalIFCapFusion(nn.Module):
    """IFCap att_gt_n_rt fusion inside MappingNetwork (768-d, residual cross-attn + FFN, no gate).

    Q' = Q + CrossAttn(LayerNorm(Q), R) + FFN(...)   (1 layer, same as IFCap)
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, num_layers: int = 1) -> None:
        super().__init__()
        self.crossatt = _IFCapCrossAttnStack(dim, num_layers=num_layers, num_heads=num_heads)

    def forward(self, q_tokens: torch.Tensor, rtf: torch.Tensor) -> torch.Tensor:
        if rtf.dim() == 2:
            rtf = rtf.unsqueeze(1)
        return self.crossatt(q_tokens, rtf)

"""DA3-to-NOVA3R token alignment modules used by VC3R."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CrossAttentionBlock(nn.Module):
    """Transformer block with target-query self-attention and source cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_source_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        query_tokens: Tensor,
        source_tokens: Tensor,
        source_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        query_norm = self.query_norm(query_tokens)
        self_attn, _ = self.self_attn(query_norm, query_norm, query_norm, need_weights=False)
        query_tokens = query_tokens + self_attn

        cross_query = self.cross_query_norm(query_tokens)
        cross_source = self.cross_source_norm(source_tokens)
        cross_attn, _ = self.cross_attn(
            cross_query,
            cross_source,
            cross_source,
            key_padding_mask=source_key_padding_mask,
            need_weights=False,
        )
        query_tokens = query_tokens + cross_attn
        query_tokens = query_tokens + self.mlp(self.mlp_norm(query_tokens))
        return query_tokens


class DA3ToNOVA3RAlignment(nn.Module):
    """Map source DA3 tokens to NOVA3R point-AE tokens.

    The module owns a fixed bank of learnable target queries shaped like the
    NOVA3R latent sequence. Each block lets those target queries attend over
    projected source tokens, then the final projection emits NOVA3R token
    channels.
    """

    def __init__(
        self,
        source_dim: int,
        hidden_dim: int = 512,
        target_tokens: int = 768,
        target_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        geom_cond: bool = False,
        geom_bands: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.target_tokens = target_tokens
        self.target_dim = target_dim
        self.source_proj = nn.Linear(source_dim, hidden_dim)
        self.target_queries = nn.Parameter(torch.randn(target_tokens, hidden_dim) * 0.02)

        # Optional explicit-geometry conditioning: a set of 3D points (same normalized
        # first-camera frame as the target tokens) is Fourier-embedded and appended to
        # the cross-attention source, so the target queries can attend over metric
        # geometry in addition to DA3 appearance features. Off by default => identical
        # to the appearance-only baseline.
        self.geom_cond = geom_cond
        if geom_cond:
            self.geom_bands = geom_bands
            self.geom_embed = nn.Sequential(
                nn.Linear(3 * 2 * geom_bands, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # learned per-stream markers so the queries can tell appearance from geometry
            self.modality_embed = nn.Parameter(torch.zeros(2, hidden_dim))
            nn.init.normal_(self.modality_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, target_dim)

    def _fourier(self, xyz: Tensor) -> Tensor:
        """NeRF-style positional encoding of xyz -> [..., 3 * 2 * geom_bands]."""
        freqs = (2.0 ** torch.arange(self.geom_bands, device=xyz.device, dtype=xyz.dtype)) * math.pi
        scaled = xyz[..., None] * freqs                       # [..., 3, bands]
        emb = torch.cat([scaled.sin(), scaled.cos()], dim=-1)  # [..., 3, 2*bands]
        return emb.reshape(*xyz.shape[:-1], -1)

    def forward(
        self,
        source_tokens: Tensor,
        geom_xyz: Tensor | None = None,
        source_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if source_tokens.ndim != 3:
            raise ValueError(f"source_tokens must be shaped [B, N, C], got {tuple(source_tokens.shape)}")

        source_tokens = self.source_proj(source_tokens)
        if self.geom_cond and geom_xyz is not None:
            if geom_xyz.ndim != 3 or geom_xyz.shape[-1] != 3:
                raise ValueError(f"geom_xyz must be shaped [B, M, 3], got {tuple(geom_xyz.shape)}")
            geom_tokens = self.geom_embed(self._fourier(geom_xyz))          # [B, M, H]
            source_tokens = source_tokens + self.modality_embed[0]
            geom_tokens = geom_tokens + self.modality_embed[1]
            source_tokens = torch.cat([source_tokens, geom_tokens], dim=1)
            if source_key_padding_mask is not None:
                pad = torch.zeros(geom_tokens.shape[:2], dtype=torch.bool, device=geom_tokens.device)
                source_key_padding_mask = torch.cat([source_key_padding_mask, pad], dim=1)
        query_tokens = self.target_queries.unsqueeze(0).expand(source_tokens.shape[0], -1, -1)
        for block in self.blocks:
            query_tokens = block(
                query_tokens,
                source_tokens,
                source_key_padding_mask=source_key_padding_mask,
            )
        return self.out_proj(self.norm(query_tokens))

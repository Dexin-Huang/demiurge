"""
Relational JEPA Predictor

Graph transformer over object slots with edge features from pairwise geometry.
Supports multi-horizon prediction and C-JEPA-style object masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


class EdgeMessageMLP(nn.Module):
    """Message function between object pairs: psi(s_i, s_j, r_ij)."""

    def __init__(self, slot_dim: int, pairwise_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * slot_dim + pairwise_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim),
        )

    def forward(self, s_i: Tensor, s_j: Tensor, r_ij: Tensor) -> Tensor:
        """
        Args:
            s_i: (B, K, K, D_slot) source slot states (broadcast)
            s_j: (B, K, K, D_slot) target slot states (broadcast)
            r_ij: (B, K, K, D_pair) pairwise features

        Returns:
            messages: (B, K, K, D_slot)
        """
        return self.net(torch.cat([s_i, s_j, r_ij], dim=-1))


class GraphTransformerBlock(nn.Module):
    """Single graph transformer block with edge-conditioned attention."""

    def __init__(self, slot_dim: int, pairwise_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = slot_dim // num_heads
        assert slot_dim % num_heads == 0

        self.norm1 = nn.LayerNorm(slot_dim)
        self.norm2 = nn.LayerNorm(slot_dim)

        self.to_qkv = nn.Linear(slot_dim, 3 * slot_dim, bias=False)
        self.edge_proj = nn.Linear(pairwise_dim, num_heads)  # edge bias per head
        self.out_proj = nn.Linear(slot_dim, slot_dim)

        self.ffn = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(slot_dim * 4, slot_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, slots: Tensor, pairwise: Tensor) -> Tensor:
        """
        Args:
            slots: (B, K, D_slot)
            pairwise: (B, K, K, D_pair)

        Returns:
            slots: (B, K, D_slot)
        """
        B, K, D = slots.shape

        # Self-attention with edge bias
        x = self.norm1(slots)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b k (h d) -> b h k d", h=self.num_heads) for t in qkv]

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Add edge bias: (B, K, K, D_pair) -> (B, K, K, H) -> (B, H, K, K)
        edge_bias = self.edge_proj(pairwise).permute(0, 3, 1, 2)
        attn = attn + edge_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h k d -> b k (h d)")
        out = self.out_proj(out)
        slots = slots + out

        # FFN
        slots = slots + self.ffn(self.norm2(slots))
        return slots


class RelationalPredictor(nn.Module):
    """Graph transformer predictor over object slots.

    Predicts future slot states at multiple horizons.
    Supports C-JEPA-style object masking.
    """

    def __init__(
        self,
        slot_dim: int,
        pairwise_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        prediction_horizons: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.prediction_horizons = prediction_horizons

        self.blocks = nn.ModuleList([
            GraphTransformerBlock(slot_dim, pairwise_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Shared prediction head with horizon conditioning
        self.horizon_embed = nn.Embedding(max(prediction_horizons) + 1, slot_dim)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
        )

    def forward(
        self,
        slot_states: Tensor,
        pairwise: Tensor,
        horizons: tuple[int, ...] | None = None,
        object_mask: Tensor | None = None,
    ) -> dict[int, Tensor]:
        """
        Args:
            slot_states: (B, K, D_slot) current slot states
            pairwise: (B, K, K, D_pair) pairwise geometric features
            horizons: which horizons to predict (default: self.prediction_horizons)
            object_mask: (B, K) binary mask — 0 = masked object whose future
                must be inferred from context (C-JEPA style)

        Returns:
            predictions: dict mapping horizon h -> (B, K, D_slot) predicted states
        """
        horizons = horizons or self.prediction_horizons

        # Apply object masking if provided
        if object_mask is not None:
            slot_states = slot_states * object_mask.unsqueeze(-1)

        # Process through graph transformer
        x = slot_states
        for block in self.blocks:
            x = block(x, pairwise)

        # Predict at each horizon
        predictions = {}
        for h in horizons:
            h_emb = self.horizon_embed(
                torch.tensor(h, device=x.device).expand(x.shape[0], x.shape[1])
            )
            predictions[h] = self.pred_head(x + h_emb)

        return predictions

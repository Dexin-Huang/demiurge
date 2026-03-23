"""
Object Tokenizer

Converts dense LeWM video tokens into K tracked object slots + background.
For initial experiments: uses simulator masks or tracked proposals.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ObjectTokenizer(nn.Module):
    """Base class for converting dense features into object slots."""

    def __init__(self, input_dim: int, slot_dim: int, num_slots: int = 8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.slot_proj = nn.Linear(input_dim, slot_dim)

    def forward(self, dense_features: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError


class SimTokenizer(ObjectTokenizer):
    """Tokenizer using ground-truth simulator masks.

    For environments where object identity and segmentation are known
    (Push-T, pymunk sim, PHYRE). Pools dense patch tokens within each
    object's mask region.
    """

    def __init__(self, input_dim: int, slot_dim: int, num_slots: int = 8):
        super().__init__(input_dim, slot_dim, num_slots)

    def forward(
        self,
        dense_features: Tensor,
        object_masks: Tensor | None = None,
        object_features: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            dense_features: (B, D) CLS token embedding from LeWM encoder
            object_masks: (B, K, H, W) binary masks per object (optional)
            object_features: (B, K, D_obj) precomputed per-object features
                from simulator state (alternative to mask pooling)

        Returns:
            slots: (B, K, D_slot)
        """
        if object_features is not None:
            # Direct projection from simulator-provided per-object features
            return self.slot_proj(object_features)

        if object_masks is not None:
            # Broadcast CLS embedding to all slots, modulated by mask presence
            B, K = object_masks.shape[:2]
            mask_present = object_masks.flatten(2).any(dim=-1).float()  # (B, K)
            # Expand CLS to all slots
            expanded = dense_features.unsqueeze(1).expand(B, K, -1)  # (B, K, D)
            slots = self.slot_proj(expanded)
            # Zero out slots for absent objects
            slots = slots * mask_present.unsqueeze(-1)
            return slots

        # Fallback: split CLS embedding into K equal chunks
        B, D = dense_features.shape
        K = self.num_slots
        chunk_size = D // K
        chunks = dense_features[:, : K * chunk_size].view(B, K, chunk_size)
        return self.slot_proj(chunks)

"""
Layer 3 — Nurture (Flexible Context)

Per-slot context vector updated with standard gradient descent.
No special constraints. Assembles the final slot state.
"""

import torch
import torch.nn as nn
from torch import Tensor


class FlexibleContext(nn.Module):
    """Flexible per-slot context — appearance, texture, task-specific features.

    Projects LeWM slot embeddings into context vectors and assembles
    the final slot state: s_it = [c_it, p_i, g_tilde_it]
    """

    def __init__(self, input_dim: int, context_dim: int = 128):
        super().__init__()
        self.context_dim = context_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(self, slot_embedding: Tensor) -> Tensor:
        """Project slot embedding into context vector.

        Args:
            slot_embedding: (B, K, D_in) from LeWM / tokenizer

        Returns:
            context: (B, K, D_ctx)
        """
        return self.projection(slot_embedding)

    @staticmethod
    def assemble_slot_state(
        context: Tensor,
        gated_unary: Tensor,
        property_memory: Tensor | None = None,
    ) -> Tensor:
        """Assemble final slot state: s_it = [c_it, p_i, g_tilde_it]

        Args:
            context: (B, K, D_ctx) flexible context
            gated_unary: (B, K, D_unary) gated geometric features
            property_memory: (B, K, D_prop) or None

        Returns:
            slot_state: (B, K, D_slot)
        """
        parts = [context, gated_unary]
        if property_memory is not None:
            parts.append(property_memory)
        return torch.cat(parts, dim=-1)

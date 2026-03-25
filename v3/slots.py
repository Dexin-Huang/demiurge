"""
Slot Attention + Static/Dynamic Decomposition

Extracts K object slots from ViT patch tokens,
then splits each slot into:
  - static: appearance code (carried forward, not predicted)
  - dynamic: (q, v) physical state (predicted by Interaction Network)
"""

import torch
import torch.nn as nn
from torch import Tensor


class SlotAttention(nn.Module):
    """Slot Attention (Locatello et al., 2020) over ViT patch tokens.

    K slots compete for N patch tokens via iterative attention.
    Each slot learns to bind to one object (or background).
    """

    def __init__(
        self,
        input_dim: int = 192,
        slot_dim: int = 128,
        num_slots: int = 3,
        n_iters: int = 3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters

        # Learnable slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        # Project patch tokens
        self.project_input = nn.Linear(input_dim, slot_dim)
        self.norm_input = nn.LayerNorm(slot_dim)

        # Attention
        self.project_q = nn.Linear(slot_dim, slot_dim)
        self.project_k = nn.Linear(slot_dim, slot_dim)
        self.project_v = nn.Linear(slot_dim, slot_dim)

        # Update
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, patch_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            patch_tokens: (B, N, D) ViT patch tokens (excluding CLS)

        Returns:
            slots: (B, K, D_slot) slot embeddings
            attn_weights: (B, K, N) attention weights (for visualization)
        """
        B, N, _ = patch_tokens.shape
        K = self.num_slots

        inputs = self.norm_input(self.project_input(patch_tokens))

        # Initialize with noise for symmetry breaking
        slots = self.slot_mu + self.slot_log_sigma.exp() * torch.randn(
            B, K, self.slot_dim, device=patch_tokens.device
        )

        attn_weights = None
        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)    # (B, K, D)
            k = self.project_k(inputs)   # (B, N, D)
            v = self.project_v(inputs)   # (B, N, D)

            # Attention: softmax over SLOTS (they compete for patches)
            attn = torch.einsum("bkd,bnd->bkn", q, k) / (self.slot_dim ** 0.5)
            attn_weights = attn.softmax(dim=1)  # (B, K, N) — normalized over K

            # Weighted sum
            updates = torch.einsum("bkn,bnd->bkd", attn_weights, v)

            # GRU + MLP residual
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, K, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_weights


class SlotDecomposer(nn.Module):
    """Decomposes slots into static (appearance) and dynamic (physics) parts.

    Static: appearance code — color, shape, texture. Constant over time.
    Dynamic: physical state — (q_x, q_y, v_x, v_y). Changes with physics.
    """

    def __init__(self, slot_dim: int = 128, static_dim: int = 64, state_dim: int = 4):
        super().__init__()
        self.static_dim = static_dim
        self.state_dim = state_dim

        # Static head: slot → appearance code
        self.static_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, static_dim),
        )

        # Dynamic head: slot → (q_x, q_y, v_x, v_y)
        self.dynamic_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, state_dim),
        )

        # Dynamic encoder: (q, v) → dynamic embedding
        # Used to reassemble predicted state back into slot space
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(state_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim - static_dim),
        )

    def decompose(self, slots: Tensor) -> dict[str, Tensor]:
        """Split slots into static and dynamic components.

        Args:
            slots: (B, K, D_slot)

        Returns:
            static: (B, K, D_static) appearance codes
            state: (B, K, 4) physical state (q_x, q_y, v_x, v_y)
        """
        return {
            "static": self.static_head(slots),
            "state": self.dynamic_head(slots),
        }

    def assemble(self, static: Tensor, state: Tensor) -> Tensor:
        """Reassemble predicted state + carried static into slot embedding.

        Args:
            static: (B, K, D_static) appearance (from current frame)
            state: (B, K, 4) predicted (q', v')

        Returns:
            slot: (B, K, D_slot) reassembled slot
        """
        dynamic_emb = self.dynamic_encoder(state)
        return torch.cat([static, dynamic_emb], dim=-1)

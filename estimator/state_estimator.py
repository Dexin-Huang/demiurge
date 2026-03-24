"""
State Estimator — Part 1

Large encoder that infers from pixels:
  - Object discovery and tracking
  - Approximate 3D geometry and contact cues
  - Initial belief over hidden material properties (z, σ)

Uses frozen LeWM encoder as the perceptual backbone.
"""

import torch
import torch.nn as nn
from torch import Tensor


class StateEstimator(nn.Module):
    """Extracts structured physical state from LeWM embeddings.

    Takes dense visual embeddings and produces per-object state:
        s_i = (q_i, v_i, z_i, σ_i)

    where:
        q_i: position/orientation (estimated from visual features)
        v_i: velocity (finite difference or optical flow)
        z_i: material code (initial estimate, refined by simulator)
        σ_i: uncertainty over material code
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_slots: int = 8,
        state_dim: int = 4,       # (x, y) position + (vx, vy) velocity
        material_dim: int = 8,    # tiny material code
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.state_dim = state_dim
        self.material_dim = material_dim

        # Slot extraction from dense embeddings
        self.slot_attention = SlotExtractor(
            input_dim=embed_dim,
            slot_dim=hidden_dim,
            num_slots=num_slots,
        )

        # Per-slot heads
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),  # q, v
        )

        self.material_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, material_dim),  # z_i
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, material_dim),  # σ_i (log-scale)
            nn.Softplus(),  # ensure positive
        )

    def forward(
        self,
        embeddings: Tensor,
        object_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Extract structured state from visual embeddings.

        Args:
            embeddings: (B, D) LeWM CLS token embeddings
            object_features: (B, K, D_obj) optional GT object features

        Returns:
            dict with:
                slots: (B, K, D_slot)
                q_v: (B, K, state_dim) position and velocity
                z: (B, K, material_dim) material code
                sigma: (B, K, material_dim) uncertainty
        """
        if object_features is not None:
            slots = object_features
        else:
            slots = self.slot_attention(embeddings)

        q_v = self.geometry_head(slots)
        z = self.material_head(slots)
        sigma = self.uncertainty_head(slots)

        return {
            "slots": slots,
            "q_v": q_v,
            "z": z,
            "sigma": sigma,
        }


class SlotExtractor(nn.Module):
    """Extract K object slots from dense visual features.

    For ground-truth experiments: uses provided object features.
    For learned experiments: simplified slot attention.
    """

    def __init__(self, input_dim: int, slot_dim: int, num_slots: int = 8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Learnable slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)

        # Attention over input
        self.project_input = nn.Linear(input_dim, slot_dim)
        self.project_slots = nn.Linear(slot_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, D) dense features

        Returns:
            slots: (B, K, D_slot)
        """
        B = x.shape[0]

        # Initialize slots
        slots = self.slot_mu.expand(B, -1, -1)

        # Project input — broadcast CLS to spatial dimension
        inputs = self.project_input(x).unsqueeze(1)  # (B, 1, D)
        inputs = self.norm_input(inputs)

        # Iterative attention (3 iterations)
        for _ in range(3):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention: slots attend to input
            q = self.project_slots(slots)  # (B, K, D)
            attn = torch.einsum("bkd,bnd->bkn", q, inputs)  # (B, K, 1)
            attn = attn / (self.slot_dim ** 0.5)
            attn = attn.softmax(dim=1)

            updates = torch.einsum("bkn,bnd->bkd", attn, inputs)  # (B, K, D)

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)

        return slots

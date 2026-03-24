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
        patch_tokens: Tensor,
        object_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Extract structured state from visual features.

        Args:
            patch_tokens: (B, N, D) spatial patch tokens from ViT encoder
                          (excluding CLS). Use encoder output[:, 1:] to get these.
            object_features: (B, K, D_obj) optional GT object features
                             (bypasses slot attention for ground-truth experiments)

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
            slots = self.slot_attention(patch_tokens)

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
    """Extract K object slots from dense visual features via slot attention.

    Requires spatial patch tokens (not just CLS) to separate objects.
    LeWM's ViT produces (B, N_patches+1, 192) — we use the N_patches
    spatial tokens as the input set for slot attention.
    """

    def __init__(self, input_dim: int, slot_dim: int, num_slots: int = 8, n_iters: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters

        # Learnable slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        # Project patch tokens to slot dimension
        self.project_input = nn.Linear(input_dim, slot_dim)
        self.norm_input = nn.LayerNorm(slot_dim)

        # Slot attention components
        self.project_q = nn.Linear(slot_dim, slot_dim)
        self.project_k = nn.Linear(slot_dim, slot_dim)
        self.project_v = nn.Linear(slot_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        """
        Args:
            patch_tokens: (B, N, D) spatial patch tokens from ViT encoder
                          (excluding CLS token). N = (H/patch)*(W/patch) = 256
                          for 224x224 images with patch_size=14.

        Returns:
            slots: (B, K, D_slot)
        """
        B, N, _ = patch_tokens.shape

        # Project inputs
        inputs = self.norm_input(self.project_input(patch_tokens))  # (B, N, D_slot)

        # Initialize slots with learned Gaussian
        slots = self.slot_mu + self.slot_log_sigma.exp() * torch.randn_like(
            self.slot_mu.expand(B, -1, -1)
        )

        # Iterative slot attention
        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention: slots (queries) attend to input patches (keys/values)
            q = self.project_q(slots)   # (B, K, D)
            k = self.project_k(inputs)  # (B, N, D)
            v = self.project_v(inputs)  # (B, N, D)

            # Dot-product attention
            attn = torch.einsum("bkd,bnd->bkn", q, k) / (self.slot_dim ** 0.5)

            # Normalize over slots (competition for patches)
            attn = attn.softmax(dim=1)  # (B, K, N) — slots compete

            # Weighted sum of values
            updates = torch.einsum("bkn,bnd->bkd", attn, v)  # (B, K, D)

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

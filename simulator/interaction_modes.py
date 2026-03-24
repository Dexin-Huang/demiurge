"""
Interaction Modes

Discrete pairwise modes: m_ij ∈ {none, contact, stick, slip, bonded, break}

These are NOT learned latent variables. They are discrete physical states
that determine which force law applies between a pair of objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from enum import IntEnum


class InteractionMode(IntEnum):
    """Discrete interaction modes between object pairs."""
    NONE = 0      # no interaction (far apart)
    CONTACT = 1   # normal contact (elastic/inelastic)
    STICK = 2     # sticking (friction > threshold)
    SLIP = 3      # sliding (friction below threshold)
    BONDED = 4    # rigidly connected
    BREAK = 5     # fracture / separation event

    @classmethod
    def num_modes(cls) -> int:
        return len(cls)


class ModeClassifier(nn.Module):
    """Classifies interaction mode for each object pair.

    m_ij = M(Δq_ij, Δv_ij, gap_ij, z_i, z_j)

    Uses relative geometry and material codes to determine
    which discrete interaction mode is active.
    """

    def __init__(
        self,
        state_dim: int = 2,
        material_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        # Input: Δq (D), Δv (D), gap (1), z_i (M), z_j (M)
        input_dim = state_dim * 2 + 1 + material_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, InteractionMode.num_modes()),
        )

    def forward(
        self,
        delta_q: Tensor,
        delta_v: Tensor,
        dist: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Classify interaction mode for all pairs.

        Args:
            delta_q: (B, K, K, D) relative displacement
            delta_v: (B, K, K, D) relative velocity
            dist: (B, K, K) pairwise distance
            z: (B, K, M) material codes

        Returns:
            modes: (B, K, K) integer mode labels
            logits: (B, K, K, num_modes) raw logits
        """
        B, K, _, D = delta_q.shape
        M = z.shape[-1]

        # Expand material codes to pairwise
        z_i = z.unsqueeze(2).expand(B, K, K, M)
        z_j = z.unsqueeze(1).expand(B, K, K, M)

        # Concatenate features
        features = torch.cat([
            delta_q, delta_v, dist.unsqueeze(-1), z_i, z_j
        ], dim=-1)

        logits = self.classifier(features)  # (B, K, K, num_modes)

        # Hard mode assignment (straight-through for gradients)
        modes = logits.argmax(dim=-1)  # (B, K, K)

        return modes, logits

    def mode_loss(self, logits: Tensor, target_modes: Tensor) -> Tensor:
        """Cross-entropy loss for mode classification (when GT available)."""
        return F.cross_entropy(
            logits.reshape(-1, InteractionMode.num_modes()),
            target_modes.reshape(-1).long(),
        )

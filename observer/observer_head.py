"""
Observer Head — Part 3

JEPA-style head that predicts future latent observations
FROM the simulated state, instead of learning physics itself.

The observer head's job is to close the loop:
    simulated state → predicted embedding → compare with real embedding

This is where the JEPA loss lives. The observer learns to render
the simulator's state into the same space as the encoder's output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ObserverHead(nn.Module):
    """Predicts future latent embeddings from simulated physical state.

    Maps (q, v, z) → predicted embedding in LeWM's latent space.
    Trained with JEPA loss: MSE(predicted_emb, stop_grad(actual_emb)).

    This is deliberately simple — a small MLP. The physics is in the
    simulator, not here.
    """

    def __init__(
        self,
        state_dim: int = 2,
        material_dim: int = 8,
        num_slots: int = 8,
        embed_dim: int = 192,
        hidden_dim: int = 256,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.noise_std = noise_std

        # Observer only sees (q, v) from the simulator — NOT z directly.
        # z's effect is only visible through how it changed (q, v).
        # This forces the simulator to actually use z to transform dynamics.
        slot_input = state_dim * 2  # q, v only
        self.slot_encoder = nn.Sequential(
            nn.Linear(slot_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Aggregate slots → single embedding (matching LeWM's CLS token)
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        q: Tensor,
        v: Tensor,
        z: Tensor | None = None,
    ) -> Tensor:
        """Predict latent embedding from simulated state.

        The observer only sees (q, v) — NOT z. The material code z
        influences predictions only through the simulator's dynamics.
        This prevents the observer from bypassing the simulator.

        Args:
            q: (B, K, D) positions
            v: (B, K, D) velocities
            z: ignored (kept in signature for API consistency)

        Returns:
            predicted_emb: (B, embed_dim)
        """
        # Per-slot encoding — q and v only
        slot_input = torch.cat([q, v], dim=-1)  # (B, K, 2D)

        # Add noise during training to prevent memorization
        if self.training and self.noise_std > 0:
            slot_input = slot_input + self.noise_std * torch.randn_like(slot_input)

        slot_features = self.slot_encoder(slot_input)  # (B, K, H)

        # Mean-pool across slots
        pooled = slot_features.mean(dim=1)  # (B, H)

        # Project to embedding space
        return self.aggregator(pooled)  # (B, embed_dim)

    def observation_loss(
        self,
        predicted_emb: Tensor,
        target_emb: Tensor,
    ) -> Tensor:
        """JEPA-style loss: predict the latent, not the pixels.

        Args:
            predicted_emb: (B, D) from observer head
            target_emb: (B, D) from frozen LeWM encoder (stop gradient)

        Returns:
            MSE loss
        """
        return F.mse_loss(predicted_emb, target_emb.detach())

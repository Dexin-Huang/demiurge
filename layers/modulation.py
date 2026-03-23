"""
Layer 2 — Modulation (Slow Property Memory)

Gating weights learned from scene context.
Hidden property memory with slow EMA update.
"""

import torch
import torch.nn as nn
from torch import Tensor


class GatingNetwork(nn.Module):
    """Produces per-slot gates from scene context.

    Gates allow the model to suppress or amplify geometric priors
    based on task relevance.
    """

    def __init__(self, context_dim: int, gate_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.unary_gate = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gate_dim),
            nn.Sigmoid(),
        )
        self.pairwise_gate = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gate_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, scene_context: Tensor, unary_dim: int, pairwise_dim: int
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            scene_context: (B, K, D_ctx) per-slot context
            unary_dim: expected output dim for alpha
            pairwise_dim: expected output dim for beta

        Returns:
            alpha: (B, K, D_unary) unary gates
            beta: (B, K, D_pair) pairwise gates (broadcast over pairs)
        """
        alpha = self.unary_gate(scene_context)
        beta = self.pairwise_gate(scene_context)
        return alpha, beta


class PropertyMemory(nn.Module):
    """Per-object hidden property vector with slow EMA update.

    Accumulates evidence about intrinsic properties (mass, friction,
    elasticity) that are not directly observable in a single frame.

    Update rule: p_i(t) = (1 - gamma) * p_i(t-1) + gamma * f(interaction)
    """

    def __init__(
        self,
        input_dim: int,
        prop_dim: int = 32,
        gamma: float = 0.05,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.prop_dim = prop_dim
        self.gamma = gamma

        # MLP that processes interaction history into property update
        self.update_fn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prop_dim),
        )

    def forward(
        self,
        interaction_features: Tensor,
        prev_memory: Tensor | None = None,
    ) -> Tensor:
        """Update property memory with EMA.

        Args:
            interaction_features: (B, K, D_in) features from recent interactions
            prev_memory: (B, K, D_prop) previous memory state, or None for init

        Returns:
            memory: (B, K, D_prop) updated property memory
        """
        proposed = self.update_fn(interaction_features)

        if prev_memory is None:
            return proposed

        return (1 - self.gamma) * prev_memory + self.gamma * proposed

    def slow_loss(self, current: Tensor, previous: Tensor) -> Tensor:
        """Regularization loss enforcing slow change.

        L_slow = ||p_i(t) - p_i(t-1)||^2
        """
        return (current - previous).pow(2).mean()


class ModulationLayer(nn.Module):
    """Full Layer 2: gating + property memory."""

    def __init__(
        self,
        context_dim: int,
        unary_dim: int,
        pairwise_dim: int,
        prop_dim: int = 32,
        interaction_dim: int | None = None,
        gamma: float = 0.05,
    ):
        super().__init__()
        self.gating = GatingNetwork(context_dim, max(unary_dim, pairwise_dim))
        self.property_memory = PropertyMemory(
            input_dim=interaction_dim or pairwise_dim,
            prop_dim=prop_dim,
            gamma=gamma,
        )
        self.unary_dim = unary_dim
        self.pairwise_dim = pairwise_dim

    def forward(
        self,
        unary_geom: Tensor,
        pairwise_geom: Tensor,
        scene_context: Tensor,
        interaction_features: Tensor,
        prev_property_memory: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            gated_unary: (B, K, D_unary)
            gated_pairwise: (B, K, K, D_pair)
            property_memory: (B, K, D_prop)
        """
        alpha, beta = self.gating(scene_context, self.unary_dim, self.pairwise_dim)

        # Gate unary features
        gated_unary = alpha[..., : self.unary_dim] * unary_geom

        # Gate pairwise features — beta is per-slot, broadcast to pairs
        B, K, _ = beta.shape
        beta_pair = beta[..., : self.pairwise_dim]
        # Expand: (B, K, D) -> (B, K, 1, D) * (B, 1, K, D) -> mean for symmetric gate
        beta_ij = (beta_pair.unsqueeze(2) + beta_pair.unsqueeze(1)) / 2
        gated_pairwise = beta_ij * pairwise_geom

        # Update property memory
        prop_mem = self.property_memory(interaction_features, prev_property_memory)

        return gated_unary, gated_pairwise, prop_mem

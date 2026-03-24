"""
DEMIURGE v0.2 — Hybrid Belief Architecture

Three-part system:
    1. State estimator (large) — pixels → (q, v, z, σ) per object
    2. Hybrid belief simulator (small) — structured forward rollout
    3. Observer head — simulated state → predicted embedding

JEPA is the perception/training interface.
The actual physics model is the tiny simulator.
"""

import torch
import torch.nn as nn
from torch import Tensor

from estimator.state_estimator import StateEstimator
from estimator.object_tracker import ObjectTracker
from simulator.hybrid_simulator import HybridBeliefSimulator
from observer.observer_head import ObserverHead


class DemiurgeV2(nn.Module):
    """Full DEMIURGE v0.2 system.

    Forward pass:
        1. Encode frame with frozen LeWM → embedding
        2. State estimator → (q, v, z, σ) per object
        3. Simulator rolls forward N steps
        4. Observer head predicts future embedding from simulated state
        5. Loss: MSE(predicted_emb, actual_future_emb)
    """

    def __init__(
        self,
        lewm_embed_dim: int = 192,
        num_slots: int = 8,
        state_dim: int = 2,
        material_dim: int = 8,
        dt: float = 1.0 / 30.0,
        innovation_threshold: float = 0.5,
    ):
        super().__init__()

        # Part 1: State estimator
        self.estimator = StateEstimator(
            embed_dim=lewm_embed_dim,
            num_slots=num_slots,
            state_dim=state_dim * 2,  # q + v
            material_dim=material_dim,
        )

        # Object tracker (no learned params, just kinematics)
        self.tracker = ObjectTracker()

        # Part 2: Hybrid belief simulator
        self.simulator = HybridBeliefSimulator(
            state_dim=state_dim,
            material_dim=material_dim,
            dt=dt,
            innovation_threshold=innovation_threshold,
        )

        # Part 3: Observer head
        self.observer = ObserverHead(
            state_dim=state_dim,
            material_dim=material_dim,
            num_slots=num_slots,
            embed_dim=lewm_embed_dim,
        )

        self.state_dim = state_dim
        self.material_dim = material_dim
        self.num_slots = num_slots

    def forward(
        self,
        lewm_embeddings: Tensor,
        future_embeddings: Tensor | None = None,
        gt_positions: Tensor | None = None,
        gt_velocities: Tensor | None = None,
        n_rollout_steps: int = 4,
    ) -> dict:
        """Full forward pass.

        Args:
            lewm_embeddings: (B, D) frozen LeWM CLS embeddings for current frame
            future_embeddings: (B, H, D) frozen LeWM embeddings for future frames
            gt_positions: (B, K, 2) optional ground-truth positions
            gt_velocities: (B, K, 2) optional ground-truth velocities
            n_rollout_steps: number of steps to simulate forward

        Returns:
            dict with predictions, losses, state trajectory
        """
        # 1. Estimate state from visual embedding
        state = self.estimator(lewm_embeddings)
        q_v = state["q_v"]
        z = state["z"]
        sigma = state["sigma"]

        # Split q and v
        D = self.state_dim
        q = q_v[..., :D]
        v = q_v[..., D:]

        # Override with ground truth if available
        if gt_positions is not None:
            q = gt_positions
        if gt_velocities is not None:
            v = gt_velocities

        # 2. Simulate forward
        trajectory = self.simulator.rollout(
            q, v, z, sigma,
            tracker=self.tracker,
            n_steps=n_rollout_steps,
        )

        # 3. Predict future embeddings from simulated states
        predicted_embeddings = []
        for t in range(1, len(trajectory["q"])):
            pred_emb = self.observer(
                trajectory["q"][t],
                trajectory["v"][t],
                trajectory["z"][t],
            )
            predicted_embeddings.append(pred_emb)

        predicted_embeddings = torch.stack(predicted_embeddings, dim=1)  # (B, H, D)

        # 4. Compute losses
        losses = {}
        if future_embeddings is not None:
            H = min(predicted_embeddings.shape[1], future_embeddings.shape[1])
            losses["observation"] = self.observer.observation_loss(
                predicted_embeddings[:, :H],
                future_embeddings[:, :H],
            )

        # Material regularization
        losses["material_kl"] = self.simulator.material_belief.kl_loss(z, sigma)

        # Energy monitoring (not a loss, just tracking)
        with torch.no_grad():
            energies = [
                self.simulator.energy(trajectory["q"][t], trajectory["v"][t], trajectory["z"][t])
                for t in range(len(trajectory["q"]))
            ]

        return {
            "predicted_embeddings": predicted_embeddings,
            "trajectory": trajectory,
            "state": state,
            "losses": losses,
            "energies": energies,
        }

    def count_params(self) -> dict[str, int]:
        """Count parameters per component."""
        counts = {}
        for name, module in [
            ("estimator", self.estimator),
            ("simulator", self.simulator),
            ("observer", self.observer),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts

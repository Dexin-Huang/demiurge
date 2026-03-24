"""
Hybrid Belief Simulator — Part 2

Small structured core that:
  - Rolls forward (q, v) with semi-implicit Euler
  - Classifies and applies discrete pairwise interaction modes
  - Updates material belief only at mechanical innovation events
  - Maintains passivity/conservation constraints

This is NOT a learned latent predictor. It is a tiny physics engine
with learned components (force model, mode classifier, material updater).
"""

import torch
import torch.nn as nn
from torch import Tensor
from enum import IntEnum

from simulator.interaction_modes import ModeClassifier, InteractionMode
from simulator.material_belief import MaterialBelief


class HybridBeliefSimulator(nn.Module):
    """Tiny hybrid simulator over structured state.

    State per object: (q, v, z, σ)
    Pairwise modes: m_ij ∈ {none, contact, stick, slip, bonded, break}

    Dynamics:
        m_ij = M(Δq_ij, Δv_ij, gap_ij, z_i, z_j)
        v_{t+1} = v_t + dt * a_free(q, v, z) + Σ_j J_m(Δq, Δv, z_i, z_j)
        q_{t+1} = q_t + dt * v_{t+1}
        z, σ updated only when mechanical innovation > threshold
    """

    def __init__(
        self,
        state_dim: int = 2,       # position dimensions (2D)
        material_dim: int = 8,    # material code size
        force_hidden: int = 64,   # force model hidden dim
        gravity: tuple[float, ...] = (0.0, -9.81),
        dt: float = 1.0 / 30.0,
        innovation_threshold: float = 0.5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.material_dim = material_dim
        self.dt = dt
        self.innovation_threshold = innovation_threshold

        # Gravity (fixed, not learned)
        self.register_buffer(
            "gravity", torch.tensor(gravity, dtype=torch.float32)
        )

        # Free acceleration model: a_free(q, v, z, u)
        # Includes gravity + learned residual from material properties
        self.free_accel = nn.Sequential(
            nn.Linear(state_dim * 2 + material_dim, force_hidden),
            nn.GELU(),
            nn.Linear(force_hidden, state_dim),
        )

        # Per-mode impulse heads: J_m(Δq, Δv, z_i, z_j)
        # Each active mode has its own tiny force law
        impulse_input = state_dim * 2 + material_dim * 2  # Δq, Δv, z_i, z_j
        num_active_modes = InteractionMode.num_modes() - 1  # exclude NONE
        self.impulse_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(impulse_input, force_hidden),
                nn.GELU(),
                nn.Linear(force_hidden, state_dim),
            )
            for _ in range(num_active_modes)
        ])

        # Mode classifier
        self.mode_classifier = ModeClassifier(
            state_dim=state_dim,
            material_dim=material_dim,
        )

        # Material belief updater
        self.material_belief = MaterialBelief(
            material_dim=material_dim,
            state_dim=state_dim,
        )

    def step(
        self,
        q: Tensor,
        v: Tensor,
        z: Tensor,
        sigma: Tensor,
        innovation: Tensor,
        pairwise: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """One simulation step.

        Args:
            q: (B, K, D) positions
            v: (B, K, D) velocities
            z: (B, K, M) material codes
            sigma: (B, K, M) material uncertainty
            innovation: (B, K) mechanical innovation signal
            pairwise: dict with delta_q, delta_v, dist, contact

        Returns:
            dict with updated q, v, z, sigma, modes
        """
        B, K, D = q.shape

        # 1. Classify interaction modes for all pairs
        modes, mode_logits = self.mode_classifier(
            pairwise["delta_q"], pairwise["delta_v"],
            pairwise["dist"], z,
        )

        # 2. Compute free acceleration
        qv = torch.cat([q, v], dim=-1)  # (B, K, 2D)
        qvz = torch.cat([qv, z], dim=-1)  # (B, K, 2D+M)
        a_free = self.free_accel(qvz)  # (B, K, D)

        # Add gravity
        a_free = a_free + self.gravity.unsqueeze(0).unsqueeze(0)

        # 3. Compute interaction impulses per mode
        # Use Gumbel-softmax for differentiable mode selection
        mode_weights = torch.nn.functional.gumbel_softmax(
            mode_logits, tau=1.0, hard=False, dim=-1,
        )  # (B, K, K, num_modes)

        # Pairwise impulse inputs
        z_i = z.unsqueeze(2).expand(B, K, K, -1)
        z_j = z.unsqueeze(1).expand(B, K, K, -1)
        imp_input = torch.cat([
            pairwise["delta_q"], pairwise["delta_v"], z_i, z_j
        ], dim=-1)  # (B, K, K, 2D+2M)

        # Each mode head produces its impulse, weighted by soft mode selection
        # Mode 0 = NONE (skip), modes 1..5 = active modes
        impulse_per_mode = torch.stack([
            head(imp_input) for head in self.impulse_heads
        ], dim=-2)  # (B, K, K, num_active_modes, D)

        # Weight by mode probabilities (exclude NONE mode)
        active_weights = mode_weights[..., 1:]  # (B, K, K, num_active_modes)
        weighted_impulse = (impulse_per_mode * active_weights.unsqueeze(-1)).sum(dim=-2)

        # Sum over interaction partners
        impulses = weighted_impulse.sum(dim=2)  # (B, K, D)

        # 4. Semi-implicit Euler integration
        v_new = v + self.dt * a_free + impulses
        q_new = q + self.dt * v_new

        # 5. Update material belief only where innovation exceeds threshold
        z_new, sigma_new = self.material_belief.update(
            z, sigma, innovation, v, v_new, pairwise,
            threshold=self.innovation_threshold,
        )

        return {
            "q": q_new,
            "v": v_new,
            "z": z_new,
            "sigma": sigma_new,
            "modes": modes,
            "mode_logits": mode_logits,
            "impulses": impulses,
            "a_free": a_free,
        }

    def rollout(
        self,
        q0: Tensor,
        v0: Tensor,
        z0: Tensor,
        sigma0: Tensor,
        tracker,
        n_steps: int = 8,
    ) -> dict[str, list[Tensor]]:
        """Roll forward for n_steps.

        Returns trajectory as lists of tensors.
        """
        q, v, z, sigma = q0, v0, z0, sigma0
        trajectory = {"q": [q], "v": [v], "z": [z], "sigma": [sigma]}
        contact_prev = torch.zeros(q.shape[0], q.shape[1], q.shape[1], device=q.device)

        for t in range(n_steps):
            # Compute pairwise features
            pairwise = tracker.compute_pairwise(q, v)

            # Detect mechanical innovation
            innovation = tracker.detect_mechanical_innovation(
                contact_prev, pairwise["contact"],
                (v - trajectory["v"][-1]) / self.dt if t > 0 else torch.zeros_like(v),
            )

            # Step
            out = self.step(q, v, z, sigma, innovation, pairwise)
            q, v, z, sigma = out["q"], out["v"], out["z"], out["sigma"]

            trajectory["q"].append(q)
            trajectory["v"].append(v)
            trajectory["z"].append(z)
            trajectory["sigma"].append(sigma)

            contact_prev = pairwise["contact"]

        return trajectory

    def energy(self, q: Tensor, v: Tensor, z: Tensor) -> Tensor:
        """Compute total energy.

        Kinetic + potential (gravity).
        """
        # Kinetic: 0.5 * ||v||^2 (mass=1 default, z could encode mass)
        ke = 0.5 * v.pow(2).sum(dim=-1)  # (B, K)

        # Potential: -g · q (gravity dot position)
        pe = -(self.gravity.unsqueeze(0).unsqueeze(0) * q).sum(dim=-1)

        return (ke + pe).sum(dim=-1)  # (B,)

    def passivity_loss(self, energies: list[Tensor]) -> Tensor:
        """Penalize energy increase between steps.

        In a passive system, energy can only decrease (dissipation)
        or stay constant (conservation). Energy increase without
        external input violates physics.
        """
        violations = []
        for t in range(1, len(energies)):
            delta_e = energies[t] - energies[t - 1]
            # ReLU: only penalize increases, not decreases
            violations.append(torch.relu(delta_e))
        return torch.stack(violations).mean()

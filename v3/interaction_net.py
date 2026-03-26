"""
Interaction Network (Battaglia et al., 2016)

The foundational architecture for learning physics:
  - φ_R: pairwise interaction function (per edge)
  - Aggregate: sum effects per node
  - φ_O: per-object update function

Plus conservation constraints (energy, Newton 3).
Total: ~10K params. Physics is small.
"""

import torch
import torch.nn as nn
from torch import Tensor


class InteractionNetwork(nn.Module):
    """Interaction Network for object dynamics.

    Given K objects with state (q, v), computes pairwise interactions
    and predicts velocity updates. Position is integrated with
    semi-implicit Euler.

    φ_R(s_i, s_j, edge_attr) → effect_ij
    agg_i = Σ_j effect_ij
    Δv_i = φ_O(s_i, agg_i)
    v'_i = v_i + Δv_i
    q'_i = q_i + dt * v'_i
    """

    def __init__(
        self,
        state_dim: int = 4,
        edge_dim: int = 5,
        action_dim: int = 2,
        effect_dim: int = 128,    # scaled up from 32
        hidden_dim: int = 256,    # scaled up from 64
        n_message_passes: int = 2,  # 2 rounds of message passing
        dt: float = 5.0 / 60.0,
    ):
        super().__init__()
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos_dim = state_dim // 2
        self.n_message_passes = n_message_passes

        # Action projection
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.pos_dim),
        )

        # φ_R: relational model — 2 layers for expressivity
        self.phi_R = nn.Sequential(
            nn.Linear(state_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, effect_dim),
        )

        # φ_O: object model — includes action + accumulated effects
        self.phi_O = nn.Sequential(
            nn.Linear(state_dim + effect_dim + self.pos_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.pos_dim),
        )

        # For multi-pass: update edge features from updated node features
        if n_message_passes > 1:
            self.edge_update = nn.Sequential(
                nn.Linear(effect_dim + state_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, effect_dim),
            )

    def compute_edge_attr(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Compute edge attributes from object states.

        Args:
            state: (B, K, state_dim) with state = [q, v]

        Returns:
            edge_attr: (B, K, K, edge_dim) relative features
            dist: (B, K, K) pairwise distances
        """
        D = self.pos_dim
        q = state[..., :D]   # (B, K, D)
        v = state[..., D:]   # (B, K, D)

        delta_q = q.unsqueeze(2) - q.unsqueeze(1)  # (B, K, K, D)
        delta_v = v.unsqueeze(2) - v.unsqueeze(1)   # (B, K, K, D)
        dist = delta_q.norm(dim=-1, keepdim=True)    # (B, K, K, 1)

        edge_attr = torch.cat([delta_q, delta_v, dist], dim=-1)  # (B, K, K, 2D+1)
        return edge_attr, dist.squeeze(-1)

    def forward(self, state: Tensor, action: Tensor | None = None) -> dict[str, Tensor]:
        """One step of Interaction Network dynamics.

        Args:
            state: (B, K, state_dim) current state [q, v]
            action: (B, K, action_dim) per-object actions.
                    For Push-T: agent gets the action, block/bg get zeros.

        Returns:
            next_state: (B, K, state_dim) predicted next state
            effects: (B, K, K, effect_dim) pairwise interaction effects
            delta_v: (B, K, pos_dim) velocity updates
        """
        B, K, S = state.shape
        D = self.pos_dim
        q = state[..., :D]
        v = state[..., D:]

        # 1. Compute edge attributes
        edge_attr, dist = self.compute_edge_attr(state)

        # 2. Multi-pass message passing
        s_i = state.unsqueeze(2).expand(B, K, K, S)
        s_j = state.unsqueeze(1).expand(B, K, K, S)
        phi_r_input = torch.cat([s_i, s_j, edge_attr], dim=-1)
        effects = self.phi_R(phi_r_input)  # (B, K, K, effect_dim)

        for mp in range(self.n_message_passes - 1):
            # Update edge features from current effects + node states
            edge_update_input = torch.cat([effects, s_i, s_j], dim=-1)
            effects = effects + self.edge_update(edge_update_input)

        # 3. Aggregate effects per node
        agg = effects.sum(dim=2)  # (B, K, effect_dim)

        # 4. Action conditioning
        if action is not None:
            action_force = self.action_proj(action)
        else:
            action_force = torch.zeros(B, K, D, device=state.device)

        # 5. φ_O: velocity update
        phi_o_input = torch.cat([state, agg, action_force], dim=-1)
        delta_v = self.phi_O(phi_o_input)

        # 6. Semi-implicit Euler
        v_new = v + delta_v
        q_new = q + self.dt * v_new

        next_state = torch.cat([q_new, v_new], dim=-1)

        return {
            "next_state": next_state,
            "effects": effects,
            "delta_v": delta_v,
            "q": q_new,
            "v": v_new,
        }

    def energy(self, state: Tensor) -> Tensor:
        """Kinetic energy (no gravity for top-down Push-T)."""
        v = state[..., self.pos_dim:]
        return 0.5 * v.pow(2).sum(dim=(-1, -2))  # (B,)

    def energy_loss(self, states: list[Tensor]) -> Tensor:
        """Penalize energy increase between consecutive steps."""
        violations = []
        for t in range(1, len(states)):
            e_curr = self.energy(states[t])
            e_prev = self.energy(states[t - 1])
            violations.append(torch.relu(e_curr - e_prev))
        if not violations:
            return torch.tensor(0.0, device=states[0].device)
        return torch.stack(violations).mean()

    def newton3_loss(self, effects: Tensor) -> Tensor:
        """Penalize violation of Newton's 3rd law: effect_ij + effect_ji ≈ 0."""
        # effects: (B, K, K, effect_dim)
        violation = effects + effects.transpose(1, 2)  # should be ~0
        return violation.pow(2).mean()

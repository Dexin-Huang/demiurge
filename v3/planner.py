"""
Structured Planner — CEM over Slot Rollouts

Drop-in replacement for LeWM's latent CEM planning.
Matches LeWM's CEM config: 300 samples, 30 iterations, top-30 elites.

Fixes from Codex audit:
    - CEM params match LeWM (30 iters, var_scale=1.0)
    - Cost on block pose only (not all slots)
    - Agent slot identified by proximity to GT agent position
    - plan_and_compare removed (use eval_planning.py instead)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from v3.slots import hungarian_match


class StructuredPlanner:
    """CEM planner using structured slot rollouts."""

    def __init__(
        self,
        model,
        lewm,
        n_samples: int = 300,
        horizon: int = 5,
        n_iterations: int = 30,   # match LeWM's CEM config
        top_k: int = 30,
        action_dim: int = 2,
        action_range: tuple[float, float] = (-1.0, 1.0),
        var_scale: float = 1.0,   # match LeWM's CEM config
        block_slot: int = 1,      # which slot is the block (for cost)
    ):
        self.model = model
        self.lewm = lewm
        self.n_samples = n_samples
        self.horizon = horizon
        self.n_iterations = n_iterations
        self.top_k = top_k
        self.action_dim = action_dim
        self.action_range = action_range
        self.var_scale = var_scale
        self.block_slot = block_slot

        self.model.eval()
        self.lewm.eval()

    @torch.no_grad()
    def extract_patches(self, frame: Tensor) -> Tensor:
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        output = self.lewm.encoder(frame, interpolate_pos_encoding=True)
        return output.last_hidden_state[:, 1:]

    @torch.no_grad()
    def identify_agent_slot(self, state: Tensor, agent_pos: Tensor) -> int:
        """Identify which slot corresponds to the agent.

        Matches by closest position to known agent position.

        Args:
            state: (1, K, state_dim) slot states
            agent_pos: (2,) known agent position (from proprio or GT)

        Returns:
            agent_slot_idx: int
        """
        pos_dim = self.model.state_dim // 2
        slot_positions = state[0, :, :2]  # (K, 2) — first 2 dims are x, y
        dists = (slot_positions - agent_pos.unsqueeze(0)).pow(2).sum(dim=-1)
        return dists.argmin().item()

    @torch.no_grad()
    def compute_cost(
        self,
        predicted_states: list[Tensor],
        goal_state: Tensor,
    ) -> Tensor:
        """Cost = block pose distance to goal block pose.

        Only scores the BLOCK slot, not agent or background.
        Uses position + angle (first pos_dim dims).
        """
        final_state = predicted_states[-1]  # (S, K, state_dim)
        b = self.block_slot

        # Block position only (x, y) — NOT theta, which is index 2
        pred_block = final_state[:, b, :2]       # (S, 2)
        goal_block = goal_state[0, b, :2]         # (2,)

        cost = (pred_block - goal_block.unsqueeze(0)).pow(2).sum(dim=-1)  # (S,)
        return cost

    @torch.no_grad()
    def plan(
        self,
        current_frame: Tensor,
        goal_frame: Tensor,
        agent_pos: Tensor | None = None,
    ) -> Tensor:
        """Plan action sequence using CEM with structured rollouts.

        Args:
            current_frame: (1, 3, 224, 224) preprocessed
            goal_frame: (1, 3, 224, 224) preprocessed
            agent_pos: (2,) optional known agent position for slot identification

        Returns:
            best_actions: (horizon, action_dim)
        """
        device = current_frame.device
        H = self.horizon
        A = self.action_dim
        S = self.n_samples

        # Extract states
        current_patches = self.extract_patches(current_frame)
        current = self.model.extract_slots(current_patches)
        goal_patches = self.extract_patches(goal_frame)
        goal = self.model.extract_slots(goal_patches)

        # Identify agent slot if position provided
        if agent_pos is not None:
            agent_idx = self.identify_agent_slot(current["state"], agent_pos)
            self.model.agent_slot = agent_idx

        # Identify block slot via Hungarian matching to goal
        # Block is the non-agent, non-background slot closest to goal
        goal_state = goal["state"]

        # CEM
        mean = torch.zeros(H, A, device=device)
        std = torch.ones(H, A, device=device) * self.var_scale

        for iteration in range(self.n_iterations):
            noise = torch.randn(S, H, A, device=device)
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
                self.action_range[0], self.action_range[1]
            )

            # Rollout
            state = current["state"].expand(S, -1, -1).clone()
            predicted_states = []
            for t in range(H):
                dyn = self.model.predict_next(state, action=actions[:, t])
                state = dyn["next_state"]
                predicted_states.append(state)

            # Cost on block only
            costs = self.compute_cost(predicted_states, goal_state)

            # Elite update
            _, elite_idx = costs.topk(self.top_k, largest=False)
            elite_actions = actions[elite_idx]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=0.01)

        return mean

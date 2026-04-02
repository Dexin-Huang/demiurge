"""
Structured Planner — CEM over Slot Rollouts

Replaces LeWM's latent rollout with structured slot dynamics.
Same CEM budget, same action sampling — just a different world model
inside the planning loop.

LeWM plans: sample actions → rollout in CLS space → pick lowest MSE to goal CLS
We plan:    sample actions → rollout in slot/state space → pick lowest state distance to goal

Usage:
    planner = StructuredPlanner(model, lewm)
    action = planner.plan(current_frame, goal_frame, n_samples=300, horizon=5)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


class StructuredPlanner:
    """CEM planner using structured slot rollouts.

    Drop-in replacement for LeWM's CEM planning:
    same action candidates, same budget, different internal model.
    """

    def __init__(
        self,
        model,
        lewm,
        n_samples: int = 300,
        horizon: int = 5,
        n_iterations: int = 10,
        top_k: int = 30,
        action_dim: int = 2,
        action_range: tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Args:
            model: trained DemiurgeV3
            lewm: frozen LeWM encoder (for patch extraction)
            n_samples: CEM action samples per iteration
            horizon: planning horizon (steps)
            n_iterations: CEM refinement iterations
            top_k: elite samples for CEM update
            action_dim: Push-T action dim (2)
            action_range: action bounds
        """
        self.model = model
        self.lewm = lewm
        self.n_samples = n_samples
        self.horizon = horizon
        self.n_iterations = n_iterations
        self.top_k = top_k
        self.action_dim = action_dim
        self.action_range = action_range

        self.model.eval()
        self.lewm.eval()

    @torch.no_grad()
    def extract_patches(self, frame: Tensor) -> Tensor:
        """Get patch tokens from a single frame."""
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        output = self.lewm.encoder(frame, interpolate_pos_encoding=True)
        return output.last_hidden_state[:, 1:]  # (1, 256, 192)

    @torch.no_grad()
    def get_goal_state(self, goal_frame: Tensor) -> Tensor:
        """Extract goal state from goal frame."""
        patches = self.extract_patches(goal_frame)
        goal = self.model.extract_slots(patches)
        return goal["state"]  # (1, K, state_dim)

    @torch.no_grad()
    def compute_cost(
        self,
        predicted_states: list[Tensor],
        goal_state: Tensor,
    ) -> Tensor:
        """Cost = distance from final predicted state to goal state.

        Uses position components only (not velocity) for goal matching.

        Args:
            predicted_states: list of (S, K, state_dim) at each horizon step
            goal_state: (1, K, state_dim) target state

        Returns:
            costs: (S,) per-sample cost
        """
        # Use last predicted state
        final_state = predicted_states[-1]  # (S, K, state_dim)
        pos_dim = self.model.state_dim // 2

        # Compare positions only (first pos_dim dimensions)
        pred_pos = final_state[:, :, :pos_dim]  # (S, K, pos_dim)
        goal_pos = goal_state[:, :, :pos_dim].expand_as(pred_pos)  # (S, K, pos_dim)

        # Sum over objects and position dims
        cost = (pred_pos - goal_pos).pow(2).sum(dim=(-1, -2))  # (S,)
        return cost

    @torch.no_grad()
    def plan(
        self,
        current_frame: Tensor,
        goal_frame: Tensor,
    ) -> Tensor:
        """Plan action sequence using CEM with structured rollouts.

        Args:
            current_frame: (1, 3, 224, 224) current observation
            goal_frame: (1, 3, 224, 224) goal observation

        Returns:
            best_actions: (horizon, action_dim) best action sequence
        """
        device = current_frame.device
        H = self.horizon
        A = self.action_dim
        S = self.n_samples

        # Extract current and goal states
        current_patches = self.extract_patches(current_frame)
        current = self.model.extract_slots(current_patches)
        goal_state = self.get_goal_state(goal_frame)

        # CEM: initialize action distribution
        mean = torch.zeros(H, A, device=device)
        std = torch.ones(H, A, device=device) * 0.5

        for iteration in range(self.n_iterations):
            # Sample action sequences: (S, H, A)
            noise = torch.randn(S, H, A, device=device)
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
                self.action_range[0], self.action_range[1]
            )

            # Rollout each action sequence through structured dynamics
            # Expand initial state to S samples
            state = current["state"].expand(S, -1, -1).clone()  # (S, K, state_dim)
            static = current["static"].expand(S, -1, -1).clone()
            slots = current["slots"].expand(S, -1, -1).clone()

            predicted_states = []
            for t in range(H):
                dyn = self.model.predict_next(state, action=actions[:, t])
                state = dyn["next_state"]
                predicted_states.append(state)

            # Compute cost
            costs = self.compute_cost(predicted_states, goal_state)  # (S,)

            # CEM update: select top-k, refit distribution
            _, elite_idx = costs.topk(self.top_k, largest=False)
            elite_actions = actions[elite_idx]  # (top_k, H, A)
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=0.01)

        return mean  # (H, A) — best action sequence

    @torch.no_grad()
    def plan_and_compare(
        self,
        current_frame: Tensor,
        goal_frame: Tensor,
    ) -> dict:
        """Plan with both LeWM and structured model, return comparison.

        This is the key experiment: same CEM budget, different internal model.
        """
        # Structured planning
        structured_actions = self.plan(current_frame, goal_frame)

        # LeWM planning (using its native rollout)
        info = {
            "pixels": current_frame.unsqueeze(1).unsqueeze(1),  # (1, 1, 1, C, H, W)
            "goal": goal_frame.unsqueeze(1).unsqueeze(1),
        }

        # Sample same number of candidates
        S = self.n_samples
        H = self.horizon
        A = self.action_dim
        lewm_candidates = torch.randn(1, S, H, A, device=current_frame.device) * 0.5
        lewm_candidates = lewm_candidates.clamp(self.action_range[0], self.action_range[1])

        lewm_costs = self.lewm.get_cost(info, lewm_candidates)  # (1, S)
        best_lewm_idx = lewm_costs[0].argmin()
        lewm_actions = lewm_candidates[0, best_lewm_idx]  # (H, A)

        return {
            "structured_actions": structured_actions,
            "lewm_actions": lewm_actions,
        }

"""
World-Model Safety Shield

Two modes:
    1. precheck(action) — before acting, roll out and flag risk
    2. posthoc(predicted, observed) — after acting, compare prediction vs reality

The shield uses the model's OWN prediction error as the VoE signal,
not deviation from constant-velocity extrapolation.

Usage:
    shield = SafetyShield(model, lewm)

    # Before acting
    risk = shield.precheck(current_frame, action_chunk)
    if not risk.safe:
        action_chunk = shield.find_safest(current_frame, candidates)

    # After acting
    anomaly = shield.posthoc(current_frame, next_frame, action_taken)
    if anomaly.detected:
        print(f"Physics anomaly: {anomaly.reason}")
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from einops import rearrange


@dataclass
class PrecheckResult:
    """Result of pre-action safety check."""
    safe: bool
    voe_scores: list[float]       # per-step prediction surprise
    energy_violations: list[float] # per-step energy increase in block
    worst_step: int               # step with highest surprise
    rollout_states: list[Tensor]  # predicted (q, v) trajectory
    reason: str = ""


@dataclass
class PosthocResult:
    """Result of post-action anomaly detection."""
    detected: bool
    surprise: float               # ||predicted - observed||²
    predicted_state: Tensor       # what the model expected
    observed_state: Tensor        # what actually happened
    reason: str = ""


class SafetyShield:

    def __init__(
        self,
        model,
        lewm,
        voe_threshold: float = 2.0,
        energy_threshold: float = 0.5,
    ):
        self.model = model
        self.lewm = lewm
        self.voe_threshold = voe_threshold
        self.energy_threshold = energy_threshold
        self.model.eval()
        self.lewm.eval()

    @torch.no_grad()
    def _get_patches(self, frame: Tensor) -> Tensor:
        """Extract patch tokens from a frame."""
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        output = self.lewm.encoder(frame, interpolate_pos_encoding=True)
        return output.last_hidden_state[:, 1:]

    @torch.no_grad()
    def precheck(
        self,
        current_frame: Tensor,
        action_chunk: Tensor,
    ) -> PrecheckResult:
        """Roll out actions and check for risk BEFORE executing.

        Args:
            current_frame: (1, 3, 224, 224) preprocessed
            action_chunk: (T, action_dim) proposed action sequence

        Returns:
            PrecheckResult with safety verdict
        """
        patches = self._get_patches(current_frame)
        current = self.model.extract_slots(patches)
        state = current["state"]
        D = self.model.state_dim // 2

        voe_scores = []
        energy_violations = []
        states = [state]

        for t in range(action_chunk.shape[0]):
            action_t = action_chunk[t:t + 1]  # (1, action_dim)
            dyn = self.model.predict_next(state, action=action_t)
            next_state = dyn["next_state"]

            # VoE: how much does next state deviate from current?
            # High deviation with small action = suspicious
            state_change = (next_state - state).pow(2).sum().item()
            action_magnitude = action_t.pow(2).sum().item()
            # Surprise = state change normalized by action magnitude
            surprise = state_change / (action_magnitude + 0.01)
            voe_scores.append(surprise)

            # Energy check on non-agent slots only
            ke_curr = 0.5 * state[:, 1:, D:].pow(2).sum().item()  # block + bg
            ke_next = 0.5 * next_state[:, 1:, D:].pow(2).sum().item()
            energy_violations.append(max(0, ke_next - ke_curr))

            states.append(next_state)
            state = next_state

        # Verdict
        worst_step = max(range(len(voe_scores)), key=lambda i: voe_scores[i])
        reasons = []
        if max(voe_scores) > self.voe_threshold:
            reasons.append(f"VoE at step {worst_step}: {voe_scores[worst_step]:.3f}")
        if max(energy_violations) > self.energy_threshold:
            reasons.append(f"Energy violation: {max(energy_violations):.3f}")

        return PrecheckResult(
            safe=len(reasons) == 0,
            voe_scores=voe_scores,
            energy_violations=energy_violations,
            worst_step=worst_step,
            rollout_states=[s.cpu() for s in states],
            reason="; ".join(reasons) if reasons else "OK",
        )

    @torch.no_grad()
    def posthoc(
        self,
        frame_t: Tensor,
        frame_t1: Tensor,
        action: Tensor,
    ) -> PosthocResult:
        """Compare prediction vs reality AFTER acting.

        This is the real anomaly detector: did the world behave
        as the model expected?

        Args:
            frame_t: (1, 3, 224, 224) frame before action
            frame_t1: (1, 3, 224, 224) frame after action
            action: (1, action_dim) action that was taken

        Returns:
            PosthocResult with anomaly detection
        """
        patches_t = self._get_patches(frame_t)
        patches_t1 = self._get_patches(frame_t1)

        # What the model predicted
        current = self.model.extract_slots(patches_t)
        dyn = self.model.predict_next(current["state"], action=action)
        predicted_state = dyn["next_state"]

        # What actually happened
        observed = self.model.extract_slots(patches_t1)
        observed_state = observed["state"]

        # Surprise = prediction error
        surprise = (predicted_state - observed_state).pow(2).sum().item()

        reasons = []
        if surprise > self.voe_threshold:
            # Diagnose which object is surprising
            per_slot = (predicted_state - observed_state).pow(2).sum(dim=-1)[0]
            worst_slot = per_slot.argmax().item()
            slot_names = ["agent", "block", "background"]
            name = slot_names[worst_slot] if worst_slot < len(slot_names) else f"slot_{worst_slot}"
            reasons.append(f"{name} deviated: surprise={per_slot[worst_slot]:.3f}")

        return PosthocResult(
            detected=surprise > self.voe_threshold,
            surprise=surprise,
            predicted_state=predicted_state.cpu(),
            observed_state=observed_state.cpu(),
            reason="; ".join(reasons) if reasons else "OK",
        )

    @torch.no_grad()
    def find_safest(
        self,
        current_frame: Tensor,
        candidates: list[Tensor],
    ) -> tuple[int, PrecheckResult]:
        """Find safest action from candidates."""
        results = [self.precheck(current_frame, c) for c in candidates]

        safe = [(i, r) for i, r in enumerate(results) if r.safe]
        if safe:
            best_i, best_r = min(safe, key=lambda x: max(x[1].voe_scores))
        else:
            best_i = min(range(len(results)), key=lambda i: max(results[i].voe_scores))
            best_r = results[best_i]

        return best_i, best_r

    @torch.no_grad()
    def monitor_trajectory(
        self,
        frames: Tensor,
        actions: Tensor,
    ) -> list[PosthocResult]:
        """Monitor a full trajectory for anomalies frame by frame.

        Args:
            frames: (T, 3, 224, 224)
            actions: (T-1, action_dim)

        Returns:
            list of PosthocResult — one per transition
        """
        results = []
        for t in range(frames.shape[0] - 1):
            result = self.posthoc(
                frames[t:t + 1],
                frames[t + 1:t + 2],
                actions[t:t + 1],
            )
            results.append(result)
        return results

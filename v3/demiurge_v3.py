"""
DEMIURGE v0.3 — Physics-Constrained Slot Predictor

Augments LeWM with object structure and physics constraints:
    1. Slot Attention decomposes patches into object slots
    2. Slots split into static (appearance) + dynamic (q, v)
    3. Action-conditioned Interaction Network predicts future dynamics
    4. Reassemble predicted state + carried appearance → predicted slot
    5. Loss in state space + conservation constraints

Fixes from Codex review:
    - Hungarian matching for slot-object assignment
    - Action conditioning in training AND inference
    - Energy loss excludes agent slot (agent injects energy)
    - VoE = model prediction error, not constant-velocity deviation

~300K trainable params on top of frozen LeWM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from v3.slots import SlotAttention, SlotDecomposer, hungarian_match, apply_permutation
from v3.interaction_net import InteractionNetwork


class DemiurgeV3(nn.Module):

    def __init__(
        self,
        input_dim: int = 192,
        slot_dim: int = 128,
        static_dim: int = 64,
        state_dim: int = 4,
        action_dim: int = 2,
        num_slots: int = 3,
        agent_slot: int = 0,       # which slot is the agent (for energy exclusion)
        dt: float = 5.0 / 60.0,
        lambda_contrast: float = 0.1,
        lambda_energy: float = 0.1,
        lambda_newton: float = 0.1,
        lambda_state: float = 1.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_slot = agent_slot

        self.lambda_contrast = lambda_contrast
        self.lambda_energy = lambda_energy
        self.lambda_newton = lambda_newton
        self.lambda_state = lambda_state

        self.slot_attention = SlotAttention(
            input_dim=input_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
        )

        self.decomposer = SlotDecomposer(
            slot_dim=slot_dim,
            static_dim=static_dim,
            state_dim=state_dim,
        )

        self.dynamics = InteractionNetwork(
            state_dim=state_dim,
            edge_dim=state_dim + 1,
            action_dim=action_dim,
            effect_dim=128,
            hidden_dim=256,
            n_message_passes=2,
            dt=dt,
        )

    def extract_slots(
        self, patch_tokens: Tensor, prev_slots: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Extract and decompose object slots from patch tokens.

        Args:
            patch_tokens: (B, N, 192)
            prev_slots: (B, K, D_slot) predicted slots from previous frame.
                        If None, this is frame 0 (discovery mode).
        """
        slots, attn = self.slot_attention(patch_tokens, prev_slots=prev_slots)
        decomp = self.decomposer.decompose(slots)
        return {
            "slots": slots,
            "static": decomp["static"],
            "state": decomp["state"],
            "attn": attn,
        }

    def predict_next(self, state: Tensor, action: Tensor | None = None) -> dict[str, Tensor]:
        """Predict next state with action conditioning.

        Args:
            state: (B, K, 4)
            action: (B, action_dim) raw action — routed to agent slot only
        """
        per_obj_action = None
        if action is not None:
            B, K = state.shape[:2]
            per_obj_action = torch.zeros(B, K, self.action_dim, device=state.device)
            per_obj_action[:, self.agent_slot] = action
        return self.dynamics(state, action=per_obj_action)

    def forward(
        self,
        patch_tokens_t: Tensor,
        patch_tokens_t1: Tensor,
        action: Tensor | None = None,
        gt_state_t: Tensor | None = None,
        gt_state_t1: Tensor | None = None,
    ) -> dict:
        """Full forward pass with Hungarian matching and action conditioning.

        Args:
            patch_tokens_t: (B, N, 192) current frame patches
            patch_tokens_t1: (B, N, 192) next frame patches
            action: (B, action_dim) action taken between frames
            gt_state_t: (B, K_obj, 4) GT state (agent + block, no bg)
            gt_state_t1: (B, K_obj, 4) GT next state
        """
        current = self.extract_slots(patch_tokens_t)
        target = self.extract_slots(patch_tokens_t1)

        # === Hungarian matching: align predicted slots to GT objects ===
        if gt_state_t is not None:
            K_obj = gt_state_t.shape[1]  # typically 2 (agent + block)
            perm_t = hungarian_match(current["state"][:, :, :2], gt_state_t[:, :, :2])
            perm_t1 = hungarian_match(target["state"][:, :, :2], gt_state_t1[:, :, :2])

            # Reorder current slots to match GT order
            matched_state_t = apply_permutation(current["state"], perm_t)
            matched_state_t1 = apply_permutation(target["state"], perm_t1)
        else:
            matched_state_t = current["state"]
            matched_state_t1 = target["state"]
            perm_t = None

        # Predict next state with action
        dyn_out = self.predict_next(current["state"], action=action)

        # Reassemble slots
        predicted_slots = self.decomposer.assemble(
            current["static"], dyn_out["next_state"]
        )
        target_slots = target["slots"]

        # === Losses ===
        losses = {}

        # Slot loss DROPPED — it fights state loss (unordered vs Hungarian-matched)
        # Contrastive loss on matched slots only
        if perm_t is not None:
            matched_pred = apply_permutation(predicted_slots, perm_t)
            matched_tgt = apply_permutation(target_slots, perm_t)
            neg = matched_tgt[torch.randperm(matched_tgt.shape[0])]
            losses["contrast"] = self._contrastive_loss(matched_pred, matched_tgt, neg)
        else:
            neg = target_slots[torch.randperm(target_slots.shape[0])]
            losses["contrast"] = self._contrastive_loss(predicted_slots, target_slots, neg)

        # 3. Energy conservation — EXCLUDE agent slot (agent injects energy)
        losses["energy"] = self._energy_loss_no_agent(
            current["state"], dyn_out["next_state"]
        )

        # 4. Newton's 3rd law
        losses["newton3"] = self.dynamics.newton3_loss(dyn_out["effects"])

        # 5. State supervision with Hungarian matching
        if gt_state_t is not None:
            # Match predicted slots to GT, compute loss on matched pairs
            pred_matched = apply_permutation(dyn_out["next_state"], perm_t)
            losses["state_t"] = F.mse_loss(matched_state_t, gt_state_t)
            losses["state_t1"] = F.mse_loss(pred_matched[:, :K_obj], gt_state_t1)

        # Total (no slot loss — it conflicts with Hungarian-matched state loss)
        total = self.lambda_contrast * losses["contrast"]
        total = total + self.lambda_energy * losses["energy"]
        total = total + self.lambda_newton * losses["newton3"]
        if "state_t" in losses:
            total = total + self.lambda_state * losses["state_t"]
        if "state_t1" in losses:
            total = total + self.lambda_state * losses["state_t1"]

        losses["total"] = total

        return {
            "losses": losses,
            "predicted_slots": predicted_slots,
            "current": current,
            "target": target,
            "dynamics": dyn_out,
            "perm_t": perm_t,
        }

    def _energy_loss_no_agent(self, state_t: Tensor, state_t1: Tensor) -> Tensor:
        """Energy conservation excluding agent slot.

        Agent injects energy (pushing), so energy increase for agent is expected.
        Only penalize energy increase in non-agent slots (block, background).
        """
        D = self.state_dim // 2  # pos dims
        # Kinetic energy per slot
        ke_t = 0.5 * state_t[..., D:].pow(2).sum(dim=-1)    # (B, K)
        ke_t1 = 0.5 * state_t1[..., D:].pow(2).sum(dim=-1)  # (B, K)

        # Mask out agent slot
        mask = torch.ones(self.num_slots, device=state_t.device)
        mask[self.agent_slot] = 0.0

        # Penalize energy increase in non-agent slots
        delta_ke = (ke_t1 - ke_t) * mask.unsqueeze(0)
        return torch.relu(delta_ke).mean()

    def _contrastive_loss(self, pred, pos, neg, temperature=0.1):
        pred_pool = F.normalize(pred.mean(dim=1), dim=-1)
        pos_pool = F.normalize(pos.mean(dim=1), dim=-1)
        neg_pool = F.normalize(neg.mean(dim=1), dim=-1)
        pos_sim = (pred_pool * pos_pool).sum(dim=-1) / temperature
        neg_sim = (pred_pool * neg_pool).sum(dim=-1) / temperature
        logits = torch.stack([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
        return F.cross_entropy(logits, labels)

    def forward_temporal(
        self,
        patch_sequence: Tensor,
        actions: Tensor | None = None,
        gt_states: Tensor | None = None,
    ) -> dict:
        """Temporal predict-update loop over a sequence of frames.

        This is the core architecture: Kalman-style estimation where
        the Interaction Network predicts and slot attention corrects.

        Args:
            patch_sequence: (B, T, N, 192) patch tokens for T frames
            actions: (B, T-1, action_dim) actions between frames, or None
            gt_states: (B, T, K_obj, 4) GT states for supervision, or None

        Returns:
            dict with per-step corrected states, innovations, losses
        """
        B, T, N, D = patch_sequence.shape

        # Frame 0: discovery — initialize from learned prior
        corrected = self.extract_slots(patch_sequence[:, 0], prev_slots=None)
        state = corrected["state"]
        static = corrected["static"]
        slots = corrected["slots"]

        all_states = [state]
        all_innovations = []
        all_effects = []
        all_predicted_states = []
        total_state_loss = torch.tensor(0.0, device=patch_sequence.device)
        total_innovation_norm = torch.tensor(0.0, device=patch_sequence.device)

        # GT supervision for frame 0
        if gt_states is not None:
            K_obj = gt_states.shape[2]
            perm = hungarian_match(state[:, :, :2], gt_states[:, 0, :, :2])
            matched = apply_permutation(state, perm)
            total_state_loss = total_state_loss + F.mse_loss(matched[:, :K_obj], gt_states[:, 0])

        for t in range(T - 1):
            # === PREDICT: Interaction Network rolls state forward ===
            act = actions[:, t] if actions is not None else None
            dyn_out = self.predict_next(state, action=act)
            predicted_state = dyn_out["next_state"]
            all_effects.append(dyn_out["effects"])

            # Reassemble predicted state into predicted slots (gated residual)
            predicted_slots = self.decomposer.assemble(static, predicted_state, prev_slot=slots)

            # === UPDATE: Slot attention corrects prediction with observation ===
            corrected = self.extract_slots(
                patch_sequence[:, t + 1],
                prev_slots=predicted_slots,  # <-- THE KEY CHANGE
            )
            corrected_state = corrected["state"]
            static = corrected["static"]  # update appearance from new frame
            slots = corrected["slots"]

            # === INNOVATION: the surprise signal ===
            # innovation = corrected - predicted (what we saw vs what we expected)
            innovation = corrected_state - predicted_state
            innovation_norm = innovation.pow(2).sum(dim=(-1, -2))  # (B,)

            all_states.append(corrected_state)
            all_innovations.append(innovation_norm)
            all_predicted_states.append(predicted_state)
            total_innovation_norm = total_innovation_norm + innovation_norm.mean()

            # GT supervision
            if gt_states is not None:
                perm = hungarian_match(corrected_state[:, :, :2], gt_states[:, t + 1, :, :2])
                matched = apply_permutation(corrected_state, perm)
                total_state_loss = total_state_loss + F.mse_loss(matched[:, :K_obj], gt_states[:, t + 1])

                # Also supervise the prediction
                pred_matched = apply_permutation(predicted_state, perm)
                total_state_loss = total_state_loss + 0.5 * F.mse_loss(pred_matched[:, :K_obj], gt_states[:, t + 1])

            # Update state for next iteration
            state = corrected_state

        # Compute losses
        n_steps = T - 1
        losses = {}
        losses["state"] = total_state_loss / max(n_steps + 1, 1)  # +1 for frame 0
        losses["innovation"] = total_innovation_norm / max(n_steps, 1)

        # Energy conservation (non-agent slots)
        losses["energy"] = self._energy_loss_no_agent(
            all_states[0], all_states[-1]
        )

        # Newton 3
        loss_newton = torch.tensor(0.0, device=patch_sequence.device)
        for eff in all_effects:
            loss_newton = loss_newton + self.dynamics.newton3_loss(eff)
        losses["newton3"] = loss_newton / max(len(all_effects), 1)

        # Round-trip consistency: decompose(assemble(static, state)) ≈ state
        # Prevents encoder/decoder mismatch from creating false innovation
        rt_state = self.decomposer.decompose(
            self.decomposer.assemble(static, state)
        )["state"]
        losses["roundtrip"] = F.mse_loss(rt_state, state.detach())

        # Static regularization: penalize temporal change in static codes
        # Prevents appearance from leaking dynamics
        if len(all_states) > 1:
            static_init = self.decomposer.decompose(
                self.slot_attention(patch_sequence[:, 0])[0]
            )["static"]
            losses["static_reg"] = F.mse_loss(static, static_init.detach())
        else:
            losses["static_reg"] = torch.tensor(0.0, device=patch_sequence.device)

        # Total
        losses["total"] = (
            losses["state"]
            + self.lambda_energy * losses["energy"]
            + self.lambda_newton * losses["newton3"]
            + 0.1 * losses["roundtrip"]
            + 0.05 * losses["static_reg"]
        )

        return {
            "losses": losses,
            "states": all_states,
            "innovations": all_innovations,
            "predicted_states": all_predicted_states,
            "effects": all_effects,
        }

    def rollout(
        self,
        patch_tokens_init: Tensor,
        actions: Tensor | None = None,
        n_steps: int = 8,
    ) -> dict[str, list[Tensor]]:
        """Roll forward WITHOUT observations (pure prediction).

        Used for planning/shield: predict what will happen if we
        take these actions, without seeing the actual future frames.

        Args:
            patch_tokens_init: (B, N, 192) initial frame only
            actions: (B, T, action_dim) planned action sequence
            n_steps: steps to simulate
        """
        current = self.extract_slots(patch_tokens_init, prev_slots=None)
        state = current["state"]
        static = current["static"]

        trajectory = {"states": [state], "slots": [current["slots"]]}

        slots = current["slots"]
        for t in range(n_steps):
            act = actions[:, t] if actions is not None and t < actions.shape[1] else None
            dyn_out = self.predict_next(state, action=act)
            state = dyn_out["next_state"]
            pred_slot = self.decomposer.assemble(static, state, prev_slot=slots)
            slots = pred_slot

            trajectory["states"].append(state)
            trajectory["slots"].append(pred_slot)

        return trajectory

    def compute_voe(
        self,
        patch_tokens_t: Tensor,
        patch_tokens_t1: Tensor,
        action: Tensor | None = None,
        prev_slots: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute VoE as the innovation from the predict-update cycle.

        This is the mathematically correct surprise signal:
        innovation = corrected_state - predicted_state

        Args:
            patch_tokens_t: (B, N, 192) current frame
            patch_tokens_t1: (B, N, 192) next frame
            action: (B, action_dim) action taken
            prev_slots: (B, K, D_slot) slots from previous step (for tracking)

        Returns:
            voe: (B,) per-sample surprise (innovation norm)
            innovation: (B, K, 4) per-slot innovation vector
        """
        # Extract current state (with tracking if prev_slots given)
        current = self.extract_slots(patch_tokens_t, prev_slots=prev_slots)

        # PREDICT
        predicted = self.predict_next(current["state"], action=action)
        predicted_slots = self.decomposer.assemble(current["static"], predicted["next_state"])

        # UPDATE (correct prediction with observation)
        corrected = self.extract_slots(patch_tokens_t1, prev_slots=predicted_slots)

        # INNOVATION = corrected - predicted
        innovation = corrected["state"] - predicted["next_state"]
        voe = innovation.pow(2).sum(dim=(-1, -2))  # (B,)

        return voe, innovation

    def count_params(self) -> dict[str, int]:
        counts = {}
        for name, module in [
            ("slot_attention", self.slot_attention),
            ("decomposer", self.decomposer),
            ("dynamics", self.dynamics),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts

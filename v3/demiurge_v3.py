"""
DEMIURGE v0.3 — Physics-Constrained Slot Predictor

Augments LeWM with object structure and physics constraints:
    1. Slot Attention decomposes patches into object slots
    2. Slots split into static (appearance) + dynamic (q, v)
    3. Interaction Network predicts future dynamics
    4. Reassemble predicted state + carried appearance → predicted slot
    5. Loss in slot space + conservation constraints

~270K trainable params on top of frozen LeWM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from v3.slots import SlotAttention, SlotDecomposer
from v3.interaction_net import InteractionNetwork


class DemiurgeV3(nn.Module):
    """Full DEMIURGE v0.3 system.

    Frozen LeWM encoder produces patch tokens.
    Slot Attention extracts K object slots.
    Static/dynamic heads decompose each slot.
    Interaction Network predicts future dynamics.
    Loss compares predicted slots with actual future slots.
    """

    def __init__(
        self,
        input_dim: int = 192,    # LeWM patch token dim
        slot_dim: int = 128,
        static_dim: int = 64,
        state_dim: int = 4,      # (q_x, q_y, v_x, v_y)
        num_slots: int = 3,      # agent + block + background
        dt: float = 5.0 / 60.0,
        lambda_contrast: float = 0.1,
        lambda_energy: float = 0.1,
        lambda_newton: float = 0.1,
        lambda_state: float = 1.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.state_dim = state_dim

        # Loss weights
        self.lambda_contrast = lambda_contrast
        self.lambda_energy = lambda_energy
        self.lambda_newton = lambda_newton
        self.lambda_state = lambda_state

        # Slot Attention
        self.slot_attention = SlotAttention(
            input_dim=input_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
        )

        # Static/Dynamic decomposition
        self.decomposer = SlotDecomposer(
            slot_dim=slot_dim,
            static_dim=static_dim,
            state_dim=state_dim,
        )

        # Interaction Network
        self.dynamics = InteractionNetwork(
            state_dim=state_dim,
            edge_dim=state_dim + 1,  # Δq, Δv, dist
            effect_dim=32,
            hidden_dim=64,
            dt=dt,
        )

    def extract_slots(self, patch_tokens: Tensor) -> dict[str, Tensor]:
        """Extract and decompose object slots from patch tokens.

        Args:
            patch_tokens: (B, N, 192) from frozen LeWM ViT

        Returns:
            slots: (B, K, D_slot) raw slot embeddings
            static: (B, K, D_static) appearance codes
            state: (B, K, 4) physical state (q, v)
            attn: (B, K, N) attention weights
        """
        slots, attn = self.slot_attention(patch_tokens)
        decomp = self.decomposer.decompose(slots)

        return {
            "slots": slots,
            "static": decomp["static"],
            "state": decomp["state"],
            "attn": attn,
        }

    def predict_next(self, state: Tensor) -> dict[str, Tensor]:
        """Predict next physical state using Interaction Network.

        Args:
            state: (B, K, 4) current (q, v)

        Returns:
            next_state: (B, K, 4) predicted (q', v')
            effects: (B, K, K, effect_dim) interaction effects
        """
        return self.dynamics(state)

    def forward(
        self,
        patch_tokens_t: Tensor,
        patch_tokens_t1: Tensor,
        gt_state_t: Tensor | None = None,
        gt_state_t1: Tensor | None = None,
        negative_slots: Tensor | None = None,
    ) -> dict:
        """Full forward pass: extract slots, predict, compute losses.

        Args:
            patch_tokens_t: (B, N, 192) current frame patches
            patch_tokens_t1: (B, N, 192) next frame patches
            gt_state_t: (B, K, 4) optional GT state for supervision
            gt_state_t1: (B, K, 4) optional GT next state
            negative_slots: (B_neg, K, D_slot) negative examples for contrastive

        Returns:
            dict with losses and diagnostics
        """
        # Extract slots from both frames
        current = self.extract_slots(patch_tokens_t)
        target = self.extract_slots(patch_tokens_t1)

        # Predict next state from current dynamic state
        dyn_out = self.predict_next(current["state"])

        # Reassemble: static from current + predicted dynamic
        predicted_slots = self.decomposer.assemble(
            current["static"], dyn_out["next_state"]
        )
        target_slots = target["slots"]

        # === Losses ===
        losses = {}

        # 1. Slot prediction loss (primary)
        losses["slot"] = F.mse_loss(predicted_slots, target_slots.detach())

        # 2. Contrastive loss (prevents copy shortcut)
        if negative_slots is not None:
            losses["contrast"] = self._contrastive_loss(
                predicted_slots, target_slots, negative_slots
            )
        else:
            # Self-negatives: use other timesteps in batch as negatives
            # Shuffle along batch dimension
            neg = target_slots[torch.randperm(target_slots.shape[0])]
            losses["contrast"] = self._contrastive_loss(
                predicted_slots, target_slots, neg
            )

        # 3. Energy conservation
        states = [current["state"], dyn_out["next_state"]]
        losses["energy"] = self.dynamics.energy_loss(states)

        # 4. Newton's 3rd law
        losses["newton3"] = self.dynamics.newton3_loss(dyn_out["effects"])

        # 5. State supervision (when GT available)
        if gt_state_t is not None:
            losses["state_t"] = F.mse_loss(current["state"], gt_state_t)
        if gt_state_t1 is not None:
            losses["state_t1"] = F.mse_loss(dyn_out["next_state"], gt_state_t1)

        # Total loss
        total = losses["slot"]
        total = total + self.lambda_contrast * losses.get("contrast", 0)
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
        }

    def _contrastive_loss(
        self,
        pred: Tensor,
        pos: Tensor,
        neg: Tensor,
        temperature: float = 0.1,
    ) -> Tensor:
        """InfoNCE contrastive loss on slot embeddings.

        Predicted slots should be closer to true next slots than to
        random negative slots.
        """
        # Pool slots to per-scene embedding
        pred_pool = pred.mean(dim=1)  # (B, D)
        pos_pool = pos.mean(dim=1)    # (B, D)
        neg_pool = neg.mean(dim=1)    # (B, D)

        # Normalize
        pred_pool = F.normalize(pred_pool, dim=-1)
        pos_pool = F.normalize(pos_pool, dim=-1)
        neg_pool = F.normalize(neg_pool, dim=-1)

        # Positive similarity
        pos_sim = (pred_pool * pos_pool).sum(dim=-1) / temperature  # (B,)

        # Negative similarity
        neg_sim = (pred_pool * neg_pool).sum(dim=-1) / temperature  # (B,)

        # InfoNCE: -log(exp(pos) / (exp(pos) + exp(neg)))
        logits = torch.stack([pos_sim, neg_sim], dim=-1)  # (B, 2)
        labels = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
        return F.cross_entropy(logits, labels)

    def rollout(
        self,
        patch_tokens_init: Tensor,
        n_steps: int = 8,
    ) -> dict[str, list[Tensor]]:
        """Roll forward for multiple steps (inference).

        Args:
            patch_tokens_init: (B, N, 192) initial frame patches

        Returns:
            trajectory of states and predicted slots
        """
        current = self.extract_slots(patch_tokens_init)
        state = current["state"]
        static = current["static"]

        trajectory = {"states": [state], "slots": [current["slots"]]}

        for _ in range(n_steps):
            dyn_out = self.predict_next(state)
            state = dyn_out["next_state"]
            pred_slot = self.decomposer.assemble(static, state)

            trajectory["states"].append(state)
            trajectory["slots"].append(pred_slot)

        return trajectory

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

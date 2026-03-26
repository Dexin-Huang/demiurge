"""
Slot Attention + Static/Dynamic Decomposition

Extracts K object slots from ViT patch tokens,
then splits each slot into:
  - static: appearance code (carried forward, not predicted)
  - dynamic: (q, v) physical state (predicted by Interaction Network)
"""

import torch
import torch.nn as nn
from torch import Tensor
from scipy.optimize import linear_sum_assignment


class SlotAttention(nn.Module):
    """Slot Attention with Predict-Update Cycle.

    Implements the classical Kalman-style estimation loop:
        PREDICT: Interaction Network propagates slots forward
        UPDATE:  Slot attention corrects prediction with new observation

    Frame 0: slots initialized from learned prior (discovery)
    Frame t>0: slots initialized from predicted state (tracking)

    The GRU inside slot attention acts as a learned, nonlinear
    Kalman gain — fusing prediction with observation.

    When an object is occluded (no matching patches), the attention
    weights are near-uniform, the GRU update is small, and the slot
    coasts on its prediction. Object permanence for free.
    """

    def __init__(
        self,
        input_dim: int = 192,
        slot_dim: int = 128,
        num_slots: int = 3,
        n_iters: int = 3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters

        # Learnable slot initialization (used only for frame 0 — discovery)
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        # Project patch tokens
        self.project_input = nn.Linear(input_dim, slot_dim)
        self.norm_input = nn.LayerNorm(slot_dim)

        # Attention
        self.project_q = nn.Linear(slot_dim, slot_dim)
        self.project_k = nn.Linear(slot_dim, slot_dim)
        self.project_v = nn.Linear(slot_dim, slot_dim)

        # Update (GRU acts as learned Kalman gain)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(
        self,
        patch_tokens: Tensor,
        prev_slots: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            patch_tokens: (B, N, D) ViT patch tokens (excluding CLS)
            prev_slots: (B, K, D_slot) predicted slots from previous frame.
                        If None, initialize from learned prior (frame 0).

        Returns:
            slots: (B, K, D_slot) corrected slot embeddings
            attn_weights: (B, K, N) attention weights
        """
        B, N, _ = patch_tokens.shape
        K = self.num_slots

        inputs = self.norm_input(self.project_input(patch_tokens))

        # INIT: from prediction (tracking) or learned prior (discovery)
        if prev_slots is not None:
            # Tracking: initialize from predicted slots
            slots = prev_slots
        else:
            # Discovery: initialize from learned prior with noise
            slots = self.slot_mu + self.slot_log_sigma.exp() * torch.randn(
                B, K, self.slot_dim, device=patch_tokens.device
            )

        # UPDATE: iterative attention corrects the prediction
        attn_weights = None
        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)    # (B, K, D)
            k = self.project_k(inputs)   # (B, N, D)
            v = self.project_v(inputs)   # (B, N, D)

            # Attention: softmax over SLOTS (they compete for patches)
            attn = torch.einsum("bkd,bnd->bkn", q, k) / (self.slot_dim ** 0.5)
            attn_weights = attn.softmax(dim=1)  # (B, K, N) — normalized over K

            # Weighted sum of observations
            updates = torch.einsum("bkn,bnd->bkd", attn_weights, v)

            # GRU: fuses prediction (slots_prev) with observation (updates)
            # This IS the learned Kalman gain
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, K, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_weights


class SlotDecomposer(nn.Module):
    """Decomposes slots into static (appearance) and dynamic (physics) parts.

    Static: appearance code — color, shape, texture. Constant over time.
    Dynamic: physical state — (q_x, q_y, v_x, v_y). Changes with physics.
    """

    def __init__(self, slot_dim: int = 128, static_dim: int = 64, state_dim: int = 4):
        super().__init__()
        self.static_dim = static_dim
        self.state_dim = state_dim

        # Static head: slot → appearance code
        self.static_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, static_dim),
        )

        # Dynamic head: slot → (q_x, q_y, v_x, v_y)
        self.dynamic_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, state_dim),
        )

        # Dynamic encoder: (q, v) → dynamic embedding
        # Used to reassemble predicted state back into slot space
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(state_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim - static_dim),
        )

    def decompose(self, slots: Tensor) -> dict[str, Tensor]:
        """Split slots into static and dynamic components.

        Args:
            slots: (B, K, D_slot)

        Returns:
            static: (B, K, D_static) appearance codes
            state: (B, K, 4) physical state (q_x, q_y, v_x, v_y)
        """
        return {
            "static": self.static_head(slots),
            "state": self.dynamic_head(slots),
        }

    def assemble(self, static: Tensor, state: Tensor) -> Tensor:
        """Reassemble predicted state + carried static into slot embedding.

        Args:
            static: (B, K, D_static) appearance (from current frame)
            state: (B, K, 4) predicted (q', v')

        Returns:
            slot: (B, K, D_slot) reassembled slot
        """
        dynamic_emb = self.dynamic_encoder(state)
        return torch.cat([static, dynamic_emb], dim=-1)


def hungarian_match(
    predicted_state: Tensor,
    gt_state: Tensor,
) -> Tensor:
    """Match predicted slots to GT objects via Hungarian algorithm.

    Solves the assignment problem: which predicted slot corresponds
    to which GT object? Returns permutation indices.

    Args:
        predicted_state: (B, K_pred, D) predicted per-slot state
        gt_state: (B, K_gt, D) ground-truth per-object state

    Returns:
        perm: (B, K_gt) indices into predicted slots for each GT object
    """
    B = predicted_state.shape[0]
    K_gt = gt_state.shape[1]
    perms = []

    for b in range(B):
        # Cost matrix: MSE between each predicted slot and each GT object
        # (K_pred, K_gt)
        cost = torch.cdist(
            predicted_state[b].unsqueeze(0),
            gt_state[b].unsqueeze(0),
        ).squeeze(0)  # (K_pred, K_gt)

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        # row_ind[i] is the predicted slot matched to GT object col_ind[i]
        # We want: for each GT object j, which predicted slot?
        perm = torch.zeros(K_gt, dtype=torch.long, device=predicted_state.device)
        for r, c in zip(row_ind, col_ind):
            if c < K_gt:
                perm[c] = r
        perms.append(perm)

    return torch.stack(perms)  # (B, K_gt)


def apply_permutation(tensor: Tensor, perm: Tensor) -> Tensor:
    """Reorder slots according to permutation.

    Args:
        tensor: (B, K, ...) slot tensor
        perm: (B, K_gt) permutation indices

    Returns:
        reordered: (B, K_gt, ...) matched to GT order
    """
    B, K_gt = perm.shape
    batch_idx = torch.arange(B, device=tensor.device).unsqueeze(1).expand(B, K_gt)
    return tensor[batch_idx, perm]

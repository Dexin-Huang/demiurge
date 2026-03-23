"""
Training Objectives

All loss components for DEMIURGE training.
L = L_JEPA + λ₁·L_geom + λ₂·L_slow_prop + λ₃·L_gate_sparse + λ₄·L_contact
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class JEPALoss(nn.Module):
    """Main JEPA latent prediction loss.

    L_JEPA = MSE(predicted_slots, stop_gradient(target_slots))
    Averaged over all prediction horizons.
    """

    def forward(
        self, predictions: dict[int, Tensor], targets: dict[int, Tensor]
    ) -> Tensor:
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        for h in predictions:
            if h in targets:
                total = total + F.mse_loss(predictions[h], targets[h].detach())
        return total / len(predictions)


class GeometricAccuracyLoss(nn.Module):
    """Auxiliary loss ensuring scaffold features are geometrically accurate.

    Only meaningful for Condition B (learnable scaffold).
    For Condition C (frozen), scaffold is deterministic from GT so this is zero.
    """

    def forward(
        self, predicted_geom: Tensor, target_geom: Tensor
    ) -> Tensor:
        return F.mse_loss(predicted_geom, target_geom)


class SlowPropertyLoss(nn.Module):
    """Regularization enforcing slow change in property memory.

    L_slow = ||p_i(t) - p_i(t-1)||²
    """

    def forward(self, current: Tensor, previous: Tensor) -> Tensor:
        return (current - previous).pow(2).mean()


class GateSparsityLoss(nn.Module):
    """Entropy regularization to prevent gate collapse.

    Encourages gates to use their full range rather than collapsing
    to always-0 or always-1.

    L_gate = -mean(α·log(α) + (1-α)·log(1-α))
    Maximizing this encourages entropy → prevents collapse.
    We negate it because we minimize the total loss.
    """

    def forward(self, gates: Tensor) -> Tensor:
        eps = 1e-7
        gates = gates.clamp(eps, 1 - eps)
        entropy = -(gates * gates.log() + (1 - gates) * (1 - gates).log())
        # We want HIGH entropy, so return negative entropy as loss
        return -entropy.mean()


class ContactPredictionLoss(nn.Module):
    """Binary contact prediction from pairwise features.

    Contact is where geometry and hidden properties meet.
    Forces both layers to coordinate.
    """

    def __init__(self, pairwise_dim: int):
        super().__init__()
        self.contact_head = nn.Linear(pairwise_dim, 1)

    def forward(
        self, pairwise_features: Tensor, contact_labels: Tensor
    ) -> Tensor:
        """
        Args:
            pairwise_features: (B, K, K, D_pair)
            contact_labels: (B, K, K) binary

        Returns:
            BCE loss
        """
        logits = self.contact_head(pairwise_features).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, contact_labels)


class DemiurgeLoss(nn.Module):
    """Combined DEMIURGE training loss.

    L = L_JEPA + λ₁·L_geom + λ₂·L_slow + λ₃·L_gate + λ₄·L_contact

    Default weights from spec: λ₁=0.1, λ₂=0.05, λ₃=0.01, λ₄=0.1
    """

    def __init__(
        self,
        pairwise_dim: int = 0,
        lambda_geom: float = 0.1,
        lambda_slow: float = 0.05,
        lambda_gate: float = 0.01,
        lambda_contact: float = 0.1,
    ):
        super().__init__()
        self.jepa_loss = JEPALoss()
        self.geom_loss = GeometricAccuracyLoss()
        self.slow_loss = SlowPropertyLoss()
        self.gate_loss = GateSparsityLoss()
        self.contact_loss = ContactPredictionLoss(pairwise_dim) if pairwise_dim > 0 else None

        self.lambda_geom = lambda_geom
        self.lambda_slow = lambda_slow
        self.lambda_gate = lambda_gate
        self.lambda_contact = lambda_contact

    def forward(
        self,
        predictions: dict[int, Tensor],
        targets: dict[int, Tensor],
        predicted_geom: Tensor | None = None,
        target_geom: Tensor | None = None,
        current_props: Tensor | None = None,
        prev_props: Tensor | None = None,
        gates: Tensor | None = None,
        pairwise_features: Tensor | None = None,
        contact_labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all loss components.

        Returns:
            dict with 'total', 'jepa', 'geom', 'slow', 'gate', 'contact' losses
        """
        losses = {}

        # Main JEPA loss (always)
        losses["jepa"] = self.jepa_loss(predictions, targets)
        total = losses["jepa"]

        # Geometric accuracy (Condition B only)
        if predicted_geom is not None and target_geom is not None:
            losses["geom"] = self.geom_loss(predicted_geom, target_geom)
            total = total + self.lambda_geom * losses["geom"]

        # Slow property regularization (Condition E)
        if current_props is not None and prev_props is not None:
            losses["slow"] = self.slow_loss(current_props, prev_props)
            total = total + self.lambda_slow * losses["slow"]

        # Gate sparsity (Conditions D, E)
        if gates is not None:
            losses["gate"] = self.gate_loss(gates)
            total = total + self.lambda_gate * losses["gate"]

        # Contact prediction (when pairwise features available)
        if (
            self.contact_loss is not None
            and pairwise_features is not None
            and contact_labels is not None
        ):
            losses["contact"] = self.contact_loss(pairwise_features, contact_labels)
            total = total + self.lambda_contact * losses["contact"]

        losses["total"] = total
        return losses

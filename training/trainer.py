"""
DEMIURGE Trainer

Training loop with layer-specific learning rates and freeze logic.
"""

import torch
import torch.nn as nn
from pathlib import Path

from model.demiurge import DemiurgeModel, Condition
from training.objectives import DemiurgeLoss


class DemiurgeTrainer:
    """Training loop for DEMIURGE with layer-specific optimization.

    - LeWM encoder: always frozen
    - Layer 1 (scaffold): frozen for Conditions C/D/E, trainable for B
    - Layer 2 (modulation): slow learning rate
    - Layer 3 (context) + predictor: standard learning rate
    """

    def __init__(
        self,
        model: DemiurgeModel,
        loss_fn: DemiurgeLoss,
        lr: float = 1e-4,
        slow_lr_factor: float = 0.1,
        weight_decay: float = 1e-3,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device

        # Build parameter groups with layer-specific LR
        param_groups = self._build_param_groups(lr, slow_lr_factor, weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups)

        # Log parameter counts
        counts = model.count_trainable_params()
        print(f"Condition: {model.condition.value}")
        for name, count in counts.items():
            print(f"  {name}: {count:,} params")

    def _build_param_groups(
        self, lr: float, slow_lr_factor: float, weight_decay: float
    ) -> list[dict]:
        groups = []

        # Modulation layer gets slow learning rate
        if self.model.modulation is not None:
            groups.append({
                "params": list(self.model.modulation.parameters()),
                "lr": lr * slow_lr_factor,
                "weight_decay": weight_decay,
                "name": "modulation",
            })

        # Everything else (tokenizer, context, predictor) gets standard LR
        standard_params = []
        for name, module in [
            ("tokenizer", self.model.tokenizer),
            ("context", self.model.context),
            ("predictor", self.model.predictor),
        ]:
            if module is not None:
                standard_params.extend(
                    p for p in module.parameters() if p.requires_grad
                )

        if self.model.dummy_pairwise is not None:
            standard_params.extend(self.model.dummy_pairwise.parameters())

        if standard_params:
            groups.append({
                "params": standard_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "standard",
            })

        # Loss function params (contact head)
        loss_params = [p for p in self.loss_fn.parameters() if p.requires_grad]
        if loss_params:
            groups.append({
                "params": loss_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "loss_heads",
            })

        return groups

    def train_step(self, batch: dict) -> dict[str, float]:
        """Single training step.

        Args:
            batch: dict with LeWM embeddings, geometry, targets, etc.

        Returns:
            dict of loss values (detached floats)
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(
            lewm_embeddings=batch["lewm_embeddings"].to(self.device),
            positions=batch.get("positions", None),
            velocities=batch.get("velocities", None),
            accelerations=batch.get("accelerations", None),
            scales=batch.get("scales", None),
            timesteps=batch.get("timesteps", None),
            interaction_features=batch.get("interaction_features", None),
            prev_property_memory=batch.get("prev_property_memory", None),
            object_features=batch.get("object_features", None),
            cjepa_mask=batch.get("cjepa_mask", None),
        )

        # Compute losses
        losses = self.loss_fn(
            predictions=output["predictions"],
            targets=batch["targets"],
            current_props=output.get("property_memory"),
            prev_props=batch.get("prev_property_memory"),
            pairwise_features=output.get("pairwise_geom"),
            contact_labels=batch.get("contact_labels"),
        )

        # Backward
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def save_checkpoint(self, path: Path, epoch: int):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "condition": self.model.condition.value,
        }, path)

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"]

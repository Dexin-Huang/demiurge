"""
DEMIURGE — Full Stack Assembly

Wraps frozen LeWM encoder and composes the three-layer stack.
Config-driven to produce Conditions A through E.
"""

from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from layers.nature import FrozenGeometricScaffold
from layers.modulation import ModulationLayer
from layers.nurture import FlexibleContext
from model.tokenizer import SimTokenizer
from model.predictor import RelationalPredictor


class Condition(str, Enum):
    """Ablation conditions from the spec."""
    A = "lewm_baseline"       # LeWM as-is, no scaffold
    B = "learnable_geom"      # Geometric features, fully trainable
    C = "frozen_geom"         # Geometric features, frozen (Layer 1 only)
    D = "frozen_plus_gates"   # Layer 1 + Layer 2 gating
    E = "full_demiurge"       # Complete three-layer stack


class DemiurgeModel(nn.Module):
    """Full DEMIURGE model wrapping a frozen LeWM encoder.

    Architecture:
        LeWM encoder (frozen) → Object tokenizer → Layer 1/2/3 → Predictor

    The condition parameter controls which layers are active:
        A: tokenizer + predictor only (no scaffold)
        B: tokenizer + learnable scaffold + predictor
        C: tokenizer + frozen scaffold + predictor
        D: tokenizer + frozen scaffold + gating + predictor
        E: tokenizer + frozen scaffold + gating + property memory + predictor
    """

    def __init__(
        self,
        condition: Condition,
        lewm_embed_dim: int = 192,
        num_slots: int = 8,
        num_freq_bands: int = 8,
        context_dim: int = 128,
        prop_dim: int = 32,
        pred_layers: int = 4,
        pred_heads: int = 4,
        gamma: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.condition = condition

        # Object tokenizer (all conditions except A use this)
        self.tokenizer = SimTokenizer(
            input_dim=lewm_embed_dim,
            slot_dim=lewm_embed_dim,
            num_slots=num_slots,
        )

        # Layer 1 — Frozen geometric scaffold
        self.scaffold = None
        if condition in (Condition.C, Condition.D, Condition.E):
            self.scaffold = FrozenGeometricScaffold(num_freq_bands=num_freq_bands)
            # Freeze: scaffold has no params, but enforce it
            for p in self.scaffold.parameters():
                p.requires_grad = False

        # Layer 2 — Modulation (gating + property memory)
        self.modulation = None
        if condition in (Condition.D, Condition.E):
            unary_dim = self.scaffold.unary_dim
            pairwise_dim = self.scaffold.pairwise_dim
            self.modulation = ModulationLayer(
                context_dim=lewm_embed_dim,  # scene context from tokenizer
                unary_dim=unary_dim,
                pairwise_dim=pairwise_dim,
                prop_dim=prop_dim if condition == Condition.E else 0,
                gamma=gamma,
            )

        # Layer 3 — Flexible context
        self.context = FlexibleContext(
            input_dim=lewm_embed_dim,
            context_dim=context_dim,
        )

        # Compute slot dimension based on condition
        slot_dim = context_dim
        if self.scaffold is not None:
            slot_dim += self.scaffold.unary_dim
        if condition == Condition.E:
            slot_dim += prop_dim

        # Pairwise dim for predictor
        pairwise_dim = self.scaffold.pairwise_dim if self.scaffold else 0
        # If no scaffold, use a dummy pairwise dim
        if pairwise_dim == 0:
            pairwise_dim = 16
            self.dummy_pairwise = nn.Linear(lewm_embed_dim, pairwise_dim)
        else:
            self.dummy_pairwise = None

        # Predictor
        self.predictor = RelationalPredictor(
            slot_dim=slot_dim,
            pairwise_dim=pairwise_dim,
            num_layers=pred_layers,
            num_heads=pred_heads,
            dropout=dropout,
        )

        self.slot_dim = slot_dim
        self.prop_dim = prop_dim

    def forward(
        self,
        lewm_embeddings: Tensor,
        positions: Tensor | None = None,
        velocities: Tensor | None = None,
        accelerations: Tensor | None = None,
        scales: Tensor | None = None,
        timesteps: Tensor | None = None,
        interaction_features: Tensor | None = None,
        prev_property_memory: Tensor | None = None,
        object_masks: Tensor | None = None,
        object_features: Tensor | None = None,
        cjepa_mask: Tensor | None = None,
    ) -> dict:
        """Forward pass through the full stack.

        Args:
            lewm_embeddings: (B, D) frozen LeWM CLS embeddings
            positions: (B, K, 2) object positions (for scaffold)
            velocities: (B, K, 2) object velocities
            accelerations: (B, K, 2) object accelerations
            scales: (B, K, 1) object scales
            timesteps: (B, K, 1) normalized timesteps
            interaction_features: (B, K, D_int) for property memory
            prev_property_memory: (B, K, D_prop) from previous timestep
            object_masks: (B, K, H, W) for tokenizer
            object_features: (B, K, D_obj) for tokenizer
            cjepa_mask: (B, K) object-level mask for C-JEPA

        Returns:
            dict with predictions, slot_states, property_memory, etc.
        """
        # Tokenize
        slots = self.tokenizer(lewm_embeddings, object_masks=object_masks,
                               object_features=object_features)
        B, K, _ = slots.shape

        # Layer 3 — context (always active)
        context = self.context(slots)

        # Layer 1 — scaffold
        unary_geom = None
        pairwise_geom = None
        if self.scaffold is not None and positions is not None:
            unary_geom, pairwise_geom = self.scaffold(
                positions, velocities, accelerations, scales, timesteps
            )

        # Layer 2 — modulation
        property_memory = None
        if self.modulation is not None and unary_geom is not None:
            gated_unary, gated_pairwise, property_memory = self.modulation(
                unary_geom, pairwise_geom, slots, interaction_features,
                prev_property_memory,
            )
        elif unary_geom is not None:
            gated_unary = unary_geom
            gated_pairwise = pairwise_geom
        else:
            gated_unary = None
            gated_pairwise = None

        # Assemble slot state
        slot_state = FlexibleContext.assemble_slot_state(
            context, gated_unary if gated_unary is not None
            else torch.zeros(B, K, 0, device=context.device),
            property_memory,
        )

        # Pairwise features for predictor
        if gated_pairwise is not None:
            pair_feat = gated_pairwise
        elif self.dummy_pairwise is not None:
            # No scaffold — generate dummy pairwise from slot embeddings
            pair_feat = self.dummy_pairwise(slots)
            pair_feat = pair_feat.unsqueeze(2).expand(B, K, K, -1)
        else:
            pair_feat = torch.zeros(B, K, K, 16, device=context.device)

        # Predict
        predictions = self.predictor(
            slot_state, pair_feat, object_mask=cjepa_mask
        )

        return {
            "predictions": predictions,
            "slot_states": slot_state,
            "context": context,
            "unary_geom": unary_geom,
            "pairwise_geom": pairwise_geom,
            "property_memory": property_memory,
        }

    def count_trainable_params(self) -> dict[str, int]:
        """Count trainable parameters per component."""
        counts = {}
        for name, module in [
            ("tokenizer", self.tokenizer),
            ("scaffold", self.scaffold),
            ("modulation", self.modulation),
            ("context", self.context),
            ("predictor", self.predictor),
        ]:
            if module is not None:
                counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
            else:
                counts[name] = 0
        counts["total"] = sum(counts.values())
        return counts

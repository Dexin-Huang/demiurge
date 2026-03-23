"""
Violation-of-Expectation Test

Extended from LeWM's VoE paradigm (color change, teleportation) to include
physically implausible dynamics: wrong mass ratio, wrong gravity, wrong friction.

This is the single most compelling qualitative result.
"""

import torch
import numpy as np
from torch import Tensor
from dataclasses import dataclass
from scipy import stats


@dataclass
class VoEResult:
    """Result of a violation-of-expectation test."""
    violation_type: str
    plausible_surprise: np.ndarray   # (N, T) surprise over time
    implausible_surprise: np.ndarray  # (N, T) surprise over time
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool   # p < 0.01

    @property
    def mean_plausible(self) -> float:
        return float(self.plausible_surprise.mean())

    @property
    def mean_implausible(self) -> float:
        return float(self.implausible_surprise.mean())


def compute_surprise(
    model,
    slot_states_sequence: Tensor,
    pairwise_sequence: Tensor,
) -> Tensor:
    """Compute prediction error (surprise) at each timestep.

    Args:
        model: DemiurgeModel or LeWM
        slot_states_sequence: (B, T, K, D) slot states over time
        pairwise_sequence: (B, T, K, K, D_pair) pairwise features over time

    Returns:
        surprise: (B, T-1) prediction error at each step
    """
    B, T, K, D = slot_states_sequence.shape
    surprises = []

    for t in range(T - 1):
        current = slot_states_sequence[:, t]
        target = slot_states_sequence[:, t + 1]
        pairwise = pairwise_sequence[:, t]

        with torch.no_grad():
            predictions = model.predictor(current, pairwise, horizons=(1,))
            predicted = predictions[1]

        error = (predicted - target).pow(2).mean(dim=(-1, -2))  # (B,)
        surprises.append(error)

    return torch.stack(surprises, dim=1)  # (B, T-1)


def run_voe_test(
    model,
    plausible_data: dict,
    implausible_data: dict,
    violation_type: str,
) -> VoEResult:
    """Run a violation-of-expectation test.

    Args:
        model: trained model (DEMIURGE or LeWM)
        plausible_data: dict with slot_states (N, T, K, D) and pairwise (N, T, K, K, D_pair)
        implausible_data: same format, with physics violation
        violation_type: name of the violation

    Returns:
        VoEResult with statistical test
    """
    plausible_surprise = compute_surprise(
        model, plausible_data["slot_states"], plausible_data["pairwise"]
    ).cpu().numpy()

    implausible_surprise = compute_surprise(
        model, implausible_data["slot_states"], implausible_data["pairwise"]
    ).cpu().numpy()

    # Aggregate surprise per trajectory (mean over time)
    p_agg = plausible_surprise.mean(axis=1)
    i_agg = implausible_surprise.mean(axis=1)

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(i_agg, p_agg, alternative="greater")

    # Cohen's d effect size
    pooled_std = np.sqrt((p_agg.std() ** 2 + i_agg.std() ** 2) / 2)
    effect_size = (i_agg.mean() - p_agg.mean()) / (pooled_std + 1e-8)

    return VoEResult(
        violation_type=violation_type,
        plausible_surprise=plausible_surprise,
        implausible_surprise=implausible_surprise,
        p_value=float(p_value),
        effect_size=float(effect_size),
        significant=p_value < 0.01,
    )


# Violation types to test
VIOLATION_TYPES = [
    "wrong_mass_ratio",    # post-collision velocities inconsistent with mass
    "wrong_gravity",       # object falls faster/slower than expected
    "wrong_friction",      # object slides too far/little on surface
    "wrong_elasticity",    # ball bounces too high/low
    "teleportation",       # LeWM baseline test
    "color_change",        # LeWM baseline test
]

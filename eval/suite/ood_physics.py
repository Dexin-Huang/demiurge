"""
OOD Physics Perturbation Suite

Real physics changes applied to Push-T trajectories:
  - Friction change (surface becomes slippery or sticky)
  - Mass change (block becomes heavier or lighter)
  - Damping change (increased or decreased damping)
  - Actuator lag (delayed action application)
  - Gravity shift (for non-top-down envs)
  - Nuisance controls (visual changes that should NOT trigger anomaly)

Each perturbation is applied to GT state trajectories to create
"impossible" sequences that a physics-aware model should detect.
"""

import torch
import numpy as np
from torch import Tensor
from dataclasses import dataclass
from enum import Enum, auto


class PerturbationType(Enum):
    """Types of physics perturbations."""
    # Real physics changes (should be detected)
    FRICTION_HIGH = auto()    # block slides less than expected
    FRICTION_LOW = auto()     # block slides more than expected
    MASS_HEAVY = auto()       # block responds less to pushes
    MASS_LIGHT = auto()       # block responds more to pushes
    DAMPING_HIGH = auto()     # motion decays faster
    DAMPING_LOW = auto()      # motion decays slower
    ACTUATOR_LAG = auto()     # action is delayed by N frames
    TELEPORT = auto()         # object jumps to random position
    PENETRATION = auto()      # objects pass through each other

    # Nuisance controls (should NOT be detected)
    IDENTITY = auto()         # no change (control)
    NOISE_SMALL = auto()      # small observation noise


@dataclass
class PerturbedSample:
    """A trajectory with a physics perturbation applied."""
    original_state: Tensor     # (T, K, 4) original GT state
    perturbed_state: Tensor    # (T, K, 4) perturbed state
    perturbation: PerturbationType
    onset_frame: int           # frame where perturbation begins
    affected_slot: int         # which object is affected (0=agent, 1=block)
    magnitude: float           # perturbation strength


class OODPhysicsSuite:
    """Generates OOD physics perturbations for evaluation.

    Given a normal trajectory, applies various physics changes
    and returns paired (normal, perturbed) for evaluation.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def perturb(
        self,
        state: Tensor,
        perturbation: PerturbationType,
        onset_frac: float = 0.3,
        magnitude: float = 1.0,
        affected_slot: int = 1,  # block by default
    ) -> PerturbedSample:
        """Apply a physics perturbation to a trajectory.

        Args:
            state: (T, K, 4) GT state [q_x, q_y, v_x, v_y] per object
            perturbation: type of perturbation
            onset_frac: fraction of trajectory before perturbation starts
            magnitude: perturbation strength (1.0 = default)
            affected_slot: which object to perturb

        Returns:
            PerturbedSample
        """
        T, K, D = state.shape
        onset = int(T * onset_frac)
        perturbed = state.clone()

        if perturbation == PerturbationType.IDENTITY:
            pass  # no change

        elif perturbation == PerturbationType.NOISE_SMALL:
            noise = torch.randn_like(state) * 0.01 * magnitude
            perturbed = state + noise

        elif perturbation == PerturbationType.FRICTION_HIGH:
            # Block velocity decays faster (more friction)
            decay = 0.7 * magnitude
            for t in range(onset, T):
                perturbed[t, affected_slot, 2:] *= decay

        elif perturbation == PerturbationType.FRICTION_LOW:
            # Block velocity decays slower (less friction) — slides further
            boost = 1.0 + 0.3 * magnitude
            for t in range(onset, T):
                perturbed[t, affected_slot, 2:] *= boost

        elif perturbation == PerturbationType.MASS_HEAVY:
            # Block responds less to velocity changes (heavier)
            # Scale down velocity changes
            for t in range(onset + 1, T):
                dv = perturbed[t, affected_slot, 2:] - perturbed[t-1, affected_slot, 2:]
                perturbed[t, affected_slot, 2:] = perturbed[t-1, affected_slot, 2:] + dv * (0.3 / magnitude)
                # Update position from new velocity
                dt = 5.0 / 60.0
                perturbed[t, affected_slot, :2] = perturbed[t-1, affected_slot, :2] + dt * perturbed[t, affected_slot, 2:]

        elif perturbation == PerturbationType.MASS_LIGHT:
            # Block responds more to velocity changes (lighter)
            for t in range(onset + 1, T):
                dv = perturbed[t, affected_slot, 2:] - perturbed[t-1, affected_slot, 2:]
                perturbed[t, affected_slot, 2:] = perturbed[t-1, affected_slot, 2:] + dv * (3.0 * magnitude)
                dt = 5.0 / 60.0
                perturbed[t, affected_slot, :2] = perturbed[t-1, affected_slot, :2] + dt * perturbed[t, affected_slot, 2:]

        elif perturbation == PerturbationType.DAMPING_HIGH:
            # All velocities decay exponentially faster
            for t in range(onset, T):
                perturbed[t, :, 2:] *= (0.8 ** magnitude)

        elif perturbation == PerturbationType.DAMPING_LOW:
            # Velocities don't decay (no damping)
            for t in range(onset, T):
                perturbed[t, :, 2:] *= (1.05 ** magnitude)

        elif perturbation == PerturbationType.ACTUATOR_LAG:
            # Agent positions are delayed by N frames
            lag = max(1, int(2 * magnitude))
            for t in range(onset, T):
                src = max(0, t - lag)
                perturbed[t, 0] = state[src, 0]  # agent from past

        elif perturbation == PerturbationType.TELEPORT:
            # Object jumps at onset frame
            perturbed[onset, affected_slot, :2] += torch.randn(2) * 0.3 * magnitude

        elif perturbation == PerturbationType.PENETRATION:
            # Objects overlap (impossible physically)
            for t in range(onset, T):
                perturbed[t, 1, :2] = perturbed[t, 0, :2] + torch.tensor([0.01, 0.01])

        return PerturbedSample(
            original_state=state,
            perturbed_state=perturbed,
            perturbation=perturbation,
            onset_frame=onset,
            affected_slot=affected_slot,
            magnitude=magnitude,
        )

    def generate_suite(
        self,
        state: Tensor,
        magnitudes: list[float] = [0.5, 1.0, 2.0],
    ) -> list[PerturbedSample]:
        """Generate full perturbation suite from a single trajectory.

        Returns one sample per (perturbation_type, magnitude) combination.
        """
        samples = []

        # Physics perturbations (should be detected)
        physics_perturbations = [
            PerturbationType.FRICTION_HIGH,
            PerturbationType.FRICTION_LOW,
            PerturbationType.MASS_HEAVY,
            PerturbationType.MASS_LIGHT,
            PerturbationType.DAMPING_HIGH,
            PerturbationType.DAMPING_LOW,
            PerturbationType.ACTUATOR_LAG,
            PerturbationType.TELEPORT,
            PerturbationType.PENETRATION,
        ]

        for pert in physics_perturbations:
            for mag in magnitudes:
                samples.append(self.perturb(state, pert, magnitude=mag))

        # Nuisance controls (should NOT be detected)
        samples.append(self.perturb(state, PerturbationType.IDENTITY))
        for mag in magnitudes:
            samples.append(self.perturb(state, PerturbationType.NOISE_SMALL, magnitude=mag))

        return samples

    @staticmethod
    def is_physics_change(ptype: PerturbationType) -> bool:
        """Whether this perturbation is a real physics change (vs nuisance)."""
        return ptype not in (PerturbationType.IDENTITY, PerturbationType.NOISE_SMALL)

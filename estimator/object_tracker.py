"""
Object Tracker

Maintains object identity across frames.
Computes velocity from position differences.
Detects contact events for mechanical innovation signals.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ObjectTracker(nn.Module):
    """Tracks objects across time, computes kinematics.

    Given per-frame state estimates, produces:
        - Consistent object identities across frames
        - Velocity via finite difference
        - Acceleration via second-order finite difference
        - Contact/proximity flags between pairs
    """

    def __init__(self, contact_threshold: float = 0.05):
        super().__init__()
        self.contact_threshold = contact_threshold

    def compute_kinematics(
        self,
        positions: Tensor,
        dt: float = 1.0,
    ) -> dict[str, Tensor]:
        """Compute velocity and acceleration from position sequence.

        Args:
            positions: (B, T, K, 2) object positions over time
            dt: timestep

        Returns:
            dict with velocities (B, T, K, 2) and accelerations (B, T, K, 2)
        """
        B, T, K, D = positions.shape

        # Velocity: forward finite difference, pad last frame
        vel = torch.zeros_like(positions)
        vel[:, :-1] = (positions[:, 1:] - positions[:, :-1]) / dt

        # Acceleration: second-order finite difference
        acc = torch.zeros_like(positions)
        acc[:, :-1] = (vel[:, 1:] - vel[:, :-1]) / dt

        return {"velocities": vel, "accelerations": acc}

    def compute_pairwise(
        self,
        positions: Tensor,
        velocities: Tensor,
    ) -> dict[str, Tensor]:
        """Compute relative pairwise features.

        Args:
            positions: (B, K, 2)
            velocities: (B, K, 2)

        Returns:
            dict with relative displacement, distance, relative velocity,
            contact flags, closing speed
        """
        B, K, D = positions.shape

        # Relative displacement: (B, K, K, 2)
        delta_q = positions.unsqueeze(2) - positions.unsqueeze(1)

        # Distance: (B, K, K)
        dist = delta_q.norm(dim=-1)

        # Relative velocity: (B, K, K, 2)
        delta_v = velocities.unsqueeze(2) - velocities.unsqueeze(1)

        # Contact: distance below threshold
        contact = (dist < self.contact_threshold).float()

        # Closing speed: negative dot product of relative velocity with
        # displacement direction (positive = approaching)
        direction = delta_q / (dist.unsqueeze(-1) + 1e-8)
        closing_speed = -(delta_v * direction).sum(dim=-1)

        return {
            "delta_q": delta_q,
            "dist": dist,
            "delta_v": delta_v,
            "contact": contact,
            "closing_speed": closing_speed,
        }

    def detect_mechanical_innovation(
        self,
        contact_prev: Tensor,
        contact_curr: Tensor,
        acc: Tensor,
        acc_predicted: Tensor | None = None,
        threshold: float = 0.5,
    ) -> Tensor:
        """Detect events where material properties become informative.

        Mechanical innovation occurs at:
            - Contact onset (new collision)
            - Contact release
            - Unexpected acceleration (prediction error)

        Args:
            contact_prev: (B, K, K) previous contact state
            contact_curr: (B, K, K) current contact state
            acc: (B, K, 2) observed acceleration
            acc_predicted: (B, K, 2) predicted acceleration (optional)
            threshold: innovation threshold

        Returns:
            innovation: (B, K) per-object innovation signal
        """
        B, K = contact_curr.shape[:2]

        # Contact onset: was not in contact, now is
        onset = ((contact_curr - contact_prev) > 0).float().sum(dim=-1)  # (B, K)

        # Contact release
        release = ((contact_prev - contact_curr) > 0).float().sum(dim=-1)

        # Acceleration surprise
        if acc_predicted is not None:
            acc_error = (acc - acc_predicted).norm(dim=-1)  # (B, K)
        else:
            acc_error = acc.norm(dim=-1)  # use magnitude as proxy

        innovation = onset + release + acc_error
        return innovation

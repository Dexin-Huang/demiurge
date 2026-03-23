"""
Layer 1 — Nature (Frozen Geometric Scaffold)

No trainable weights. Ever.

Fixed Fourier/RBF encoding of unary and pairwise geometric features.
Deterministic and fixed at initialization.
"""

import torch
import torch.nn as nn
from torch import Tensor


class FrozenGeometricScaffold(nn.Module):
    """Frozen geometric scaffold with zero trainable parameters.

    Encodes unary per-object features (position, velocity, acceleration, scale)
    and pairwise features (relative displacement, distance, relative velocity,
    contact, depth order, time-to-contact) using fixed Fourier features.
    """

    def __init__(self, num_freq_bands: int = 8):
        super().__init__()
        self.num_freq_bands = num_freq_bands
        # Fixed frequencies — registered as buffer, NOT parameter
        freqs = 2.0 ** torch.arange(0, num_freq_bands).float() * torch.pi
        self.register_buffer("freqs", freqs)

    def encode_fourier(self, features: Tensor) -> Tensor:
        """Deterministic Fourier feature encoding.

        Args:
            features: (B, K, F) raw feature values

        Returns:
            (B, K, F * 2 * L) encoded features where L = num_freq_bands
        """
        # features: (B, K, F) -> (B, K, F, 1) * (L,) -> (B, K, F, L)
        x = features.unsqueeze(-1) * self.freqs  # broadcast
        encoded = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B, K, F, 2L)
        return encoded.flatten(-2)  # (B, K, F * 2L)

    def compute_unary(
        self,
        positions: Tensor,
        velocities: Tensor,
        accelerations: Tensor,
        scales: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """Compute Fourier-encoded unary features per object.

        Args:
            positions: (B, K, 2) normalized (x, y) position
            velocities: (B, K, 2) velocity
            accelerations: (B, K, 2) acceleration
            scales: (B, K, 1) bounding box scale
            timesteps: (B, K, 1) normalized timestep

        Returns:
            (B, K, D_unary) encoded unary features
        """
        raw = torch.cat([positions, velocities, accelerations, scales, timesteps], dim=-1)
        return self.encode_fourier(raw)

    def compute_pairwise(
        self,
        positions: Tensor,
        velocities: Tensor,
    ) -> Tensor:
        """Compute Fourier-encoded pairwise features for all object pairs.

        Args:
            positions: (B, K, 2)
            velocities: (B, K, 2)

        Returns:
            pairwise_features: (B, K, K, D_pair) encoded pairwise features
        """
        B, K, _ = positions.shape

        # Relative displacement: (B, K, K, 2)
        delta_pos = positions.unsqueeze(2) - positions.unsqueeze(1)

        # Euclidean distance: (B, K, K, 1)
        dist = delta_pos.norm(dim=-1, keepdim=True)

        # Relative velocity: (B, K, K, 2)
        delta_vel = velocities.unsqueeze(2) - velocities.unsqueeze(1)

        # Contact proxy — binary overlap when distance < threshold
        contact_threshold = 0.05  # normalized coordinates
        contact = (dist < contact_threshold).float()

        # Depth order proxy — sign of y-difference (higher y = "in front" in 2D)
        depth_order = torch.sign(delta_pos[..., 1:2])

        # Time-to-contact estimate: dist / closing_speed (clamped)
        closing_speed = -(delta_pos * delta_vel).sum(dim=-1, keepdim=True) / (dist + 1e-8)
        ttc = dist / (closing_speed.clamp(min=1e-4))
        ttc = ttc.clamp(max=10.0)  # cap at 10 normalized time units

        raw = torch.cat([delta_pos, dist, delta_vel, contact, depth_order, ttc], dim=-1)

        # Fourier encode: (B, K, K, F) -> (B, K, K, F*2L)
        x = raw.unsqueeze(-1) * self.freqs
        encoded = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return encoded.flatten(-2)

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        accelerations: Tensor,
        scales: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute frozen geometric features.

        Returns:
            unary: (B, K, D_unary)
            pairwise: (B, K, K, D_pair)
        """
        unary = self.compute_unary(positions, velocities, accelerations, scales, timesteps)
        pairwise = self.compute_pairwise(positions, velocities)
        return unary, pairwise

    @property
    def unary_dim(self) -> int:
        """Output dimension of unary features: 8 raw features * 2 * L."""
        return 8 * 2 * self.num_freq_bands

    @property
    def pairwise_dim(self) -> int:
        """Output dimension of pairwise features: 9 raw features * 2 * L."""
        return 9 * 2 * self.num_freq_bands

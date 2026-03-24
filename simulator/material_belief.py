"""
Material Belief

Per-object material code z_i and uncertainty σ_i.
Updated ONLY when mechanical innovation exceeds threshold.

"Mass doesn't change frame to frame. But your belief about mass
 should update when you see a collision."
"""

import torch
import torch.nn as nn
from torch import Tensor


class MaterialBelief(nn.Module):
    """Maintains and updates belief over hidden material properties.

    Update rule:
        if I_i >= τ:
            z_i, σ_i = U(z_i, σ_i, δ_i)
        else:
            z_i, σ_i unchanged (coast)

    where I_i is mechanical innovation (contact onset, unexpected accel, etc.)
    and δ_i is the evidence from the mechanical event.
    """

    def __init__(
        self,
        material_dim: int = 8,
        state_dim: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.material_dim = material_dim

        # Evidence encoder: extracts δ from pre/post collision state
        # Input: v_pre (D), v_post (D), z_current (M), σ_current (M)
        evidence_input = state_dim * 2 + material_dim * 2
        self.evidence_encoder = nn.Sequential(
            nn.Linear(evidence_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, material_dim * 2),  # δ_z and δ_σ
        )

    def update(
        self,
        z: Tensor,
        sigma: Tensor,
        innovation: Tensor,
        v_pre: Tensor,
        v_post: Tensor,
        pairwise: dict[str, Tensor],
        threshold: float = 0.5,
    ) -> tuple[Tensor, Tensor]:
        """Conditionally update material belief.

        Args:
            z: (B, K, M) current material codes
            sigma: (B, K, M) current uncertainty
            innovation: (B, K) innovation signal per object
            v_pre: (B, K, D) velocity before step
            v_post: (B, K, D) velocity after step
            pairwise: dict with pairwise features
            threshold: innovation threshold for update

        Returns:
            z_new: (B, K, M)
            sigma_new: (B, K, M)
        """
        # Compute evidence from velocity change
        evidence_input = torch.cat([v_pre, v_post, z, sigma], dim=-1)
        delta = self.evidence_encoder(evidence_input)
        delta_z, delta_sigma = delta.chunk(2, dim=-1)

        # Mask: only update where innovation exceeds threshold
        update_mask = (innovation > threshold).float().unsqueeze(-1)  # (B, K, 1)

        # Bayesian-ish update: reduce uncertainty, shift mean
        z_new = z + update_mask * delta_z * sigma  # scale update by uncertainty
        sigma_new = sigma * (1.0 - update_mask * 0.1)  # shrink uncertainty on update
        sigma_new = sigma_new + (1 - update_mask) * 0.001  # tiny growth when no update (prevent collapse)
        sigma_new = sigma_new.clamp(min=0.01)  # floor

        return z_new, sigma_new

    def kl_loss(self, z: Tensor, sigma: Tensor) -> Tensor:
        """KL divergence from unit Gaussian prior on material codes.

        Prevents material codes from growing unbounded.
        """
        return 0.5 * (z.pow(2) + sigma.pow(2) - sigma.log() - 1).mean()

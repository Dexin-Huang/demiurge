"""
Phase 1 End-to-End: The Full Pipeline As Designed

    Rendered frames
        → frozen LeWM encoder (patch tokens + CLS)
        → state estimator (q, v, z, σ per object)
        → hybrid simulator (rolls forward)
        → observer head (predicts future CLS embedding)
        → JEPA loss: MSE(predicted_emb, stop_grad(actual_future_emb))

This is the architecture as proposed. No shortcuts.

For Phase 1 we use ground-truth (q, v) to bypass the state estimator
and isolate whether the simulator + observer learn correct physics.
The state estimator is tested separately.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from einops import rearrange

from simulator.hybrid_simulator import HybridBeliefSimulator
from simulator.material_belief import MaterialBelief
from estimator.object_tracker import ObjectTracker
from observer.observer_head import ObserverHead
from experiments.sim2d import generate_dataset


class Phase1Model(nn.Module):
    """Phase 1 model: simulator + observer, with frozen LeWM encoder.

    Bypasses the state estimator — uses GT (q, v) from the simulator.
    The material code z and uncertainty σ are learned.

    The key: the observer only sees (q, v), not z.
    So if mass affects the trajectory, the simulator MUST use z to
    get the right (q, v), which the observer maps to the right embedding.
    """

    def __init__(
        self,
        lewm_model,
        state_dim: int = 2,
        material_dim: int = 8,
        num_objects: int = 6,
        embed_dim: int = 192,
    ):
        super().__init__()

        # Frozen LeWM encoder — only used for target embeddings
        self.lewm = lewm_model
        self.lewm.eval()
        for p in self.lewm.parameters():
            p.requires_grad = False

        self.state_dim = state_dim
        self.num_objects = num_objects
        self.embed_dim = embed_dim

        # Material code initializer: given first-frame embedding,
        # produce initial z and σ per object
        self.z_init = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_objects * material_dim),
        )
        self.sigma_init = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_objects * material_dim),
            nn.Softplus(),
        )

        # Simulator
        self.simulator = HybridBeliefSimulator(
            state_dim=state_dim,
            material_dim=material_dim,
            force_hidden=64,
            gravity=(0.0, -300.0 / 500.0),  # normalized
            dt=1.0 / 60.0,
            innovation_threshold=0.3,
        )

        # Observer — maps (q, v) to predicted LeWM embedding
        self.observer = ObserverHead(
            state_dim=state_dim,
            material_dim=material_dim,
            num_slots=num_objects,
            embed_dim=embed_dim,
            hidden_dim=256,
            noise_std=0.05,
        )

        self.tracker = ObjectTracker(contact_threshold=0.06)
        self.material_dim = material_dim

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Get LeWM CLS embeddings for a batch of frames.

        Args:
            frames: (B, T, 3, 224, 224) uint8 or float

        Returns:
            embeddings: (B, T, 192)
        """
        B, T = frames.shape[:2]
        pixels = frames.float()
        if pixels.max() > 1.0:
            pixels = pixels / 255.0

        # LeWM expects ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=pixels.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixels.device).view(1, 1, 3, 1, 1)
        pixels = (pixels - mean) / std

        flat = rearrange(pixels, "b t c h w -> (b t) c h w")
        with torch.no_grad():
            output = self.lewm.encoder(flat, interpolate_pos_encoding=True)
            cls_emb = output.last_hidden_state[:, 0]  # CLS token
            emb = self.lewm.projector(cls_emb)
        return rearrange(emb, "(b t) d -> b t d", b=B)

    def forward(
        self,
        frames: torch.Tensor,
        gt_positions: torch.Tensor,
        gt_velocities: torch.Tensor,
        rollout_len: int = 8,
    ) -> dict:
        """Full forward pass.

        Args:
            frames: (B, T, 3, 224, 224) rendered frames for T timesteps
            gt_positions: (B, T, K, 2) ground-truth positions (normalized)
            gt_velocities: (B, T, K, 2) ground-truth velocities (normalized)
            rollout_len: number of steps to simulate forward

        Returns:
            dict with losses and diagnostics
        """
        B, T, K, D = gt_positions.shape
        device = frames.device

        # 1. Encode ALL frames with frozen LeWM → target embeddings
        target_embs = self.encode_frames(frames)  # (B, T, 192)

        # 2. Initialize material codes from first-frame embedding
        first_emb = target_embs[:, 0]  # (B, 192)
        z = self.z_init(first_emb).view(B, K, self.material_dim)
        sigma = self.sigma_init(first_emb).view(B, K, self.material_dim)

        # 3. Starting state (GT)
        t0 = 0
        q = gt_positions[:, t0]  # (B, K, 2)
        v = gt_velocities[:, t0]  # (B, K, 2)

        # 4. Simulate forward and predict embeddings at each step
        predicted_embs = []
        contact_prev = torch.zeros(B, K, K, device=device)
        energies = [self.simulator.energy(q, v, z)]

        for step in range(min(rollout_len, T - 1)):
            # Compute pairwise + innovation
            pairwise = self.tracker.compute_pairwise(q, v)
            innovation = self.tracker.detect_mechanical_innovation(
                contact_prev, pairwise["contact"],
                torch.zeros_like(v) if step == 0 else (v - v_prev) / self.simulator.dt,
            )

            v_prev = v

            # Simulator step
            out = self.simulator.step(q, v, z, sigma, innovation, pairwise)
            q, v, z, sigma = out["q"], out["v"], out["z"], out["sigma"]

            # Observer: predict embedding from simulated (q, v)
            pred_emb = self.observer(q, v)  # (B, 192) — z is NOT passed
            predicted_embs.append(pred_emb)

            energies.append(self.simulator.energy(q, v, z))
            contact_prev = pairwise["contact"]

        # 5. Compute losses
        predicted_embs = torch.stack(predicted_embs, dim=1)  # (B, R, 192)
        target_future = target_embs[:, 1:rollout_len + 1]    # (B, R, 192)

        R = min(predicted_embs.shape[1], target_future.shape[1])
        loss_obs = F.mse_loss(predicted_embs[:, :R], target_future[:, :R].detach())
        loss_passivity = self.simulator.passivity_loss(energies)
        loss_kl = self.simulator.material_belief.kl_loss(z, sigma)

        total_loss = loss_obs + 0.1 * loss_passivity + 0.01 * loss_kl

        return {
            "loss": total_loss,
            "loss_obs": loss_obs.item(),
            "loss_passivity": loss_passivity.item(),
            "loss_kl": loss_kl.item(),
            "z_final": z.detach(),
            "sigma_final": sigma.detach(),
            "energies": [e.detach() for e in energies],
        }


def train(
    lewm_checkpoint: str,
    n_trajectories: int = 1000,
    n_steps: int = 30,
    n_objects: int = 6,
    rollout_len: int = 8,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 3e-4,
    device: str = "cuda",
):
    print("=== PHASE 1 E2E: Full Pipeline ===")
    print()

    # 1. Load frozen LeWM
    print("Loading LeWM encoder...")
    lewm = torch.load(lewm_checkpoint, map_location=device, weights_only=False)
    lewm.eval()
    print(f"  LeWM loaded: {sum(p.numel() for p in lewm.parameters()):,} params (frozen)")

    # 2. Generate rendered dataset
    print(f"Generating {n_trajectories} trajectories with rendering...")
    data = generate_dataset(
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        n_objects=n_objects,
        render=True,
        render_size=224,
        seed=42,
    )

    positions = data["positions"]       # (N, T+1, K, 2)
    velocities = data["velocities"]     # (N, T+1, K, 2)
    frames = data["frames"]             # (N, T+1, 3, 224, 224) uint8
    masses = data["masses"]             # (N, K)
    frictions = data["frictions"]       # (N, K)
    contacts = data["contact_matrices"]  # (N, T, K, K)

    print(f"  Frames: {frames.shape}")
    print(f"  Positions: {positions.shape}")
    print(f"  Contacts: {int(contacts.sum())} total events")
    print()

    # 3. Build model
    model = Phase1Model(
        lewm_model=lewm,
        state_dim=2,
        material_dim=8,
        num_objects=n_objects,
        embed_dim=192,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"  simulator: {sum(p.numel() for p in model.simulator.parameters()):,}")
    print(f"  observer: {sum(p.numel() for p in model.observer.parameters()):,}")
    print(f"  z_init: {sum(p.numel() for p in model.z_init.parameters()):,}")
    print(f"  sigma_init: {sum(p.numel() for p in model.sigma_init.parameters()):,}")
    print()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Train
    train_N = int(0.8 * n_trajectories)

    print("Training...")
    for epoch in range(epochs):
        model.train()
        epoch_obs, epoch_pass = [], []

        perm = torch.randperm(train_N)
        for batch_start in range(0, train_N, batch_size):
            idx = perm[batch_start:batch_start + batch_size]

            # Random start time
            max_t0 = n_steps - rollout_len - 1
            t0 = np.random.randint(0, max(1, max_t0))
            t_end = t0 + rollout_len + 1

            batch_frames = frames[idx, t0:t_end].to(device)
            batch_pos = positions[idx, t0:t_end].to(device)
            batch_vel = velocities[idx, t0:t_end].to(device)

            out = model(batch_frames, batch_pos, batch_vel, rollout_len=rollout_len)

            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_obs.append(out["loss_obs"])
            epoch_pass.append(out["loss_passivity"])

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            obs_mean = np.mean(epoch_obs)
            pass_mean = np.mean(epoch_pass)
            print(f"  Epoch {epoch+1:3d} | obs_loss={obs_mean:.4f} passivity={pass_mean:.4f}")

    # 5. Material code analysis
    print()
    print("=== MATERIAL CODE ANALYSIS ===")
    model.eval()

    all_z, all_mass, all_fric = [], [], []
    with torch.no_grad():
        for i in range(train_N, n_trajectories):
            batch_frames = frames[i:i+1, :rollout_len+1].to(device)
            batch_pos = positions[i:i+1, :rollout_len+1].to(device)
            batch_vel = velocities[i:i+1, :rollout_len+1].to(device)

            out = model(batch_frames, batch_pos, batch_vel, rollout_len=rollout_len)
            all_z.append(out["z_final"][0].cpu())
            all_mass.append(masses[i])
            all_fric.append(frictions[i])

    all_z = torch.stack(all_z).reshape(-1, 8).numpy()
    all_mass = torch.stack(all_mass).reshape(-1).numpy()
    all_fric = torch.stack(all_fric).reshape(-1).numpy()

    from sklearn.linear_model import LinearRegression
    for name, target in [("mass", all_mass), ("friction", all_fric)]:
        reg = LinearRegression()
        reg.fit(all_z, target)
        pred = reg.predict(all_z)
        r, _ = pearsonr(pred, target)
        print(f"  {name:>10}: r={r:.3f}")

    print()
    print("=== PHASE 1 E2E COMPLETE ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/runpod-volume/pusht/lejepa_object.ckpt")
    parser.add_argument("--n_traj", type=int, default=500)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--n_objects", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(
        lewm_checkpoint=args.checkpoint,
        n_trajectories=args.n_traj,
        n_steps=args.n_steps,
        n_objects=args.n_objects,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

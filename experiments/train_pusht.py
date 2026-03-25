"""
Push-T Integration: DEMIURGE v0.2 on Real Data

Full pipeline as designed:
    Push-T frames → frozen LeWM encoder (CLS + patches)
    → state estimator (slot attention on patches → q, v, z, σ)
    → hybrid simulator (rolls forward block dynamics)
    → observer head (predicts future CLS embedding)
    → JEPA loss

For Phase 1: uses GT agent positions, only simulates block response.
This isolates whether the simulator learns contact/push dynamics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines", "lewm"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from einops import rearrange
import time
import argparse

from simulator.hybrid_simulator import HybridBeliefSimulator
from estimator.object_tracker import ObjectTracker
from observer.observer_head import ObserverHead


def encode_with_patches(lewm, pixels):
    """Extract CLS embeddings AND patch tokens from frozen LeWM.

    Args:
        lewm: frozen JEPA model
        pixels: (B, T, 3, 224, 224) normalized float tensors

    Returns:
        cls_embs: (B, T, 192) projected CLS embeddings (target for JEPA loss)
        patch_tokens: (B, T, 256, 192) raw ViT patch tokens (input for slot attention)
    """
    B, T = pixels.shape[:2]
    flat = rearrange(pixels, "b t c h w -> (b t) c h w")

    with torch.no_grad():
        output = lewm.encoder(flat, interpolate_pos_encoding=True)
        all_tokens = output.last_hidden_state  # (B*T, 257, 192)

        cls_raw = all_tokens[:, 0]      # (B*T, 192)
        patches = all_tokens[:, 1:]     # (B*T, 256, 192)

        # CLS goes through projector for target embedding
        cls_emb = lewm.projector(cls_raw)  # (B*T, 192)

    cls_embs = rearrange(cls_emb, "(b t) d -> b t d", b=B)
    patch_tokens = rearrange(patches, "(b t) n d -> b t n d", b=B)
    return cls_embs, patch_tokens


def parse_gt_state(state, dt=5.0 / 60.0):
    """Parse Push-T state vector into per-object (q, v).

    State layout: [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]

    Args:
        state: (B, T, 7) raw state from dataset (may be normalized)
        dt: timestep for block velocity finite difference

    Returns:
        agent_q: (B, T, 2) agent positions
        agent_v: (B, T, 2) agent velocities
        block_q: (B, T, 2) block positions
        block_v: (B, T, 2) block velocities (from finite difference)
    """
    agent_q = state[..., :2]     # (B, T, 2)
    block_q = state[..., 2:4]    # (B, T, 2)
    agent_v = state[..., 5:7]    # (B, T, 2)

    # Block velocity via finite difference
    block_v = torch.zeros_like(block_q)
    block_v[:, :-1] = (block_q[:, 1:] - block_q[:, :-1]) / dt

    return agent_q, agent_v, block_q, block_v


class PushTModel(nn.Module):
    """DEMIURGE v0.2 adapted for Push-T.

    Key differences from generic DemiurgeV2:
        - No gravity (top-down task)
        - K=2 objects (agent + block)
        - GT agent positions used; only block is simulated
        - Action not modeled explicitly — agent motion is observed
    """

    def __init__(self, lewm, embed_dim=192, material_dim=8):
        super().__init__()

        self.lewm = lewm
        self.lewm.eval()
        for p in self.lewm.parameters():
            p.requires_grad = False

        K = 2  # agent + block
        self.K = K
        self.embed_dim = embed_dim
        self.material_dim = material_dim

        # Simulator — no gravity for top-down Push-T
        self.simulator = HybridBeliefSimulator(
            state_dim=2,
            material_dim=material_dim,
            force_hidden=64,
            gravity=(0.0, 0.0),  # top-down, no gravity
            dt=5.0 / 60.0,
            innovation_threshold=0.3,
        )

        # Observer — maps simulated (q, v) → predicted CLS embedding
        self.observer = ObserverHead(
            state_dim=2,
            material_dim=material_dim,
            num_slots=K,
            embed_dim=embed_dim,
            hidden_dim=256,
            noise_std=0.05,
        )

        # Material code initializer from CLS embedding
        self.z_init = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, K * material_dim),
        )
        self.sigma_init = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, K * material_dim),
            nn.Softplus(),
        )

        self.tracker = ObjectTracker(contact_threshold=0.05)

    def forward(
        self,
        cls_embs: torch.Tensor,
        agent_q_seq: torch.Tensor,
        agent_v_seq: torch.Tensor,
        block_q_init: torch.Tensor,
        block_v_init: torch.Tensor,
        rollout_len: int = 4,
    ) -> dict:
        """Forward pass: simulate block dynamics, predict future embeddings.

        Args:
            cls_embs: (B, T, 192) target CLS embeddings from LeWM
            agent_q_seq: (B, T, 2) GT agent positions for full sequence
            agent_v_seq: (B, T, 2) GT agent velocities
            block_q_init: (B, 2) initial block position
            block_v_init: (B, 2) initial block velocity
            rollout_len: steps to simulate

        Returns:
            dict with losses and diagnostics
        """
        B = cls_embs.shape[0]
        device = cls_embs.device

        # Initialize material codes from first CLS embedding
        first_emb = cls_embs[:, 0]  # (B, 192)
        z = self.z_init(first_emb).view(B, self.K, self.material_dim)
        sigma = self.sigma_init(first_emb).view(B, self.K, self.material_dim)

        # Initial state: stack agent + block into K=2 objects
        q = torch.stack([agent_q_seq[:, 0], block_q_init], dim=1)  # (B, 2, 2)
        v = torch.stack([agent_v_seq[:, 0], block_v_init], dim=1)  # (B, 2, 2)

        # Rollout
        predicted_embs = []
        contact_prev = torch.zeros(B, self.K, self.K, device=device)
        energies = [self.simulator.energy(q, v, z)]

        for step in range(min(rollout_len, cls_embs.shape[1] - 1)):
            # Compute pairwise features + innovation
            pairwise = self.tracker.compute_pairwise(q, v)
            innovation = self.tracker.detect_mechanical_innovation(
                contact_prev, pairwise["contact"],
                torch.zeros_like(v) if step == 0 else (v - v_prev) / self.simulator.dt,
            )
            v_prev = v.clone()

            # Simulator step
            out = self.simulator.step(q, v, z, sigma, innovation, pairwise)

            # Agent: use GT next position (not simulated)
            # Block: use simulated position
            next_agent_q = agent_q_seq[:, step + 1]  # (B, 2)
            next_agent_v = agent_v_seq[:, step + 1]  # (B, 2)
            sim_block_q = out["q"][:, 1]              # (B, 2)
            sim_block_v = out["v"][:, 1]              # (B, 2)

            q = torch.stack([next_agent_q, sim_block_q], dim=1)
            v = torch.stack([next_agent_v, sim_block_v], dim=1)
            z, sigma = out["z"], out["sigma"]

            # Observer predicts embedding from (q, v)
            pred_emb = self.observer(q, v)
            predicted_embs.append(pred_emb)

            energies.append(self.simulator.energy(q, v, z))
            contact_prev = pairwise["contact"]

        # Losses
        predicted_embs = torch.stack(predicted_embs, dim=1)  # (B, R, 192)
        target_embs = cls_embs[:, 1:rollout_len + 1]         # (B, R, 192)

        R = min(predicted_embs.shape[1], target_embs.shape[1])
        loss_obs = F.mse_loss(predicted_embs[:, :R], target_embs[:, :R].detach())
        loss_passivity = self.simulator.passivity_loss(energies)
        loss_kl = self.simulator.material_belief.kl_loss(z, sigma)

        total = loss_obs + 0.1 * loss_passivity + 0.01 * loss_kl

        return {
            "loss": total,
            "loss_obs": loss_obs.item(),
            "loss_passivity": loss_passivity.item(),
            "loss_kl": loss_kl.item(),
            "z_final": z.detach(),
        }


def train(
    checkpoint: str,
    data_dir: str = "/runpod-volume",
    rollout_len: int = 4,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda",
    num_workers: int = 4,
):
    print("=== DEMIURGE v0.2 on Push-T ===")
    print()

    # 1. Load frozen LeWM
    print("Loading LeWM...")
    lewm = torch.load(checkpoint, map_location=device, weights_only=False)
    lewm.eval()
    print(f"  Params: {sum(p.numel() for p in lewm.parameters()):,} (frozen)")

    # 2. Load Push-T dataset
    print("Loading dataset...")
    import stable_worldmodel as swm
    from utils import get_img_preprocessor, get_column_normalizer
    import stable_pretraining as spt

    seq_len = 1 + rollout_len  # need context + future frames
    ds = swm.data.HDF5Dataset(
        name="pusht_expert_train",
        num_steps=seq_len,
        frameskip=5,
        keys_to_load=["pixels", "action", "proprio", "state"],
        keys_to_cache=["action", "proprio", "state"],
        cache_dir=data_dir,
    )

    # Transforms: image preprocessing + state normalization
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
    for col in ["action", "proprio", "state"]:
        transforms.append(get_column_normalizer(ds, col, col))
    ds.transform = spt.data.transforms.Compose(*transforms)

    print(f"  Samples: {len(ds)}")

    # Train/val split
    n_train = int(0.9 * len(ds))
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(ds)))

    train_ds = torch.utils.data.Subset(ds, train_indices)
    val_ds = torch.utils.data.Subset(ds, val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # 3. Build model
    model = PushTModel(lewm, embed_dim=192, material_dim=8).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"    simulator: {sum(p.numel() for p in model.simulator.parameters()):,}")
    print(f"    observer:  {sum(p.numel() for p in model.observer.parameters()):,}")
    print(f"    z_init:    {sum(p.numel() for p in model.z_init.parameters()):,}")
    print()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Train
    print("Training...")
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_obs = []
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            pixels = batch["pixels"].to(device)   # (B, T, 3, 224, 224)
            state = batch["state"].to(device)      # (B, T, 7) normalized

            # Encode with frozen LeWM
            cls_embs, _ = encode_with_patches(lewm, pixels)

            # Parse GT state
            agent_q, agent_v, block_q, block_v = parse_gt_state(state)

            # Forward
            out = model(
                cls_embs=cls_embs,
                agent_q_seq=agent_q,
                agent_v_seq=agent_v,
                block_q_init=block_q[:, 0],
                block_v_init=block_v[:, 0],
                rollout_len=rollout_len,
            )

            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_obs.append(out["loss_obs"])

            if batch_idx >= 200:  # cap batches per epoch for speed
                break

        scheduler.step()
        elapsed = time.time() - t0

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vi, vbatch in enumerate(val_loader):
                    vpix = vbatch["pixels"].to(device)
                    vstate = vbatch["state"].to(device)
                    vcls, _ = encode_with_patches(lewm, vpix)
                    aq, av, bq, bv = parse_gt_state(vstate)
                    vout = model(vcls, aq, av, bq[:, 0], bv[:, 0], rollout_len)
                    val_losses.append(vout["loss_obs"])
                    if vi >= 50:
                        break

            val_loss = np.mean(val_losses)
            train_loss = np.mean(epoch_obs)
            marker = " *" if val_loss < best_val else ""
            if val_loss < best_val:
                best_val = val_loss
            print(f"  Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f} | {elapsed:.0f}s{marker}")

    print()
    print(f"Best val obs_loss: {best_val:.4f}")
    print()

    # 5. Baseline comparison: what does a trivial predictor achieve?
    print("=== BASELINE: copy-current-embedding predictor ===")
    copy_losses = []
    with torch.no_grad():
        for vi, vbatch in enumerate(val_loader):
            vpix = vbatch["pixels"].to(device)
            vcls, _ = encode_with_patches(lewm, vpix)
            # Trivial: predict future = current
            trivial_pred = vcls[:, :rollout_len].expand_as(vcls[:, 1:rollout_len+1])
            R = min(trivial_pred.shape[1], vcls[:, 1:rollout_len+1].shape[1])
            copy_loss = F.mse_loss(trivial_pred[:, :R], vcls[:, 1:rollout_len+1, :R].detach())
            copy_losses.append(copy_loss.item())
            if vi >= 50:
                break
    print(f"  Copy-current baseline: {np.mean(copy_losses):.4f}")
    print(f"  Our model:             {best_val:.4f}")
    print()
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/runpod-volume/pusht/lejepa_object.ckpt")
    parser.add_argument("--data_dir", default="/runpod-volume")
    parser.add_argument("--rollout_len", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        rollout_len=args.rollout_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

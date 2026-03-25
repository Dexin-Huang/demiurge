"""
Push-T Training: DEMIURGE v0.3 — Physics-Constrained Slot Predictor

Full pipeline:
    Push-T frames → frozen LeWM ViT → patch tokens
    → Slot Attention → K object slots
    → Static/Dynamic decomposition
    → Interaction Network predicts future dynamic state
    → Reassemble → compare with actual future slots
    → Slot MSE + contrastive + conservation losses

Uses GT state for supervision when available.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines", "lewm"))

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import time
import argparse


def extract_patches(lewm, pixels):
    """Get patch tokens from frozen LeWM encoder.

    Args:
        pixels: (B, T, 3, 224, 224) preprocessed frames

    Returns:
        patches: (B, T, 256, 192) ViT patch tokens (excluding CLS)
    """
    B, T = pixels.shape[:2]
    flat = rearrange(pixels, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        output = lewm.encoder(flat, interpolate_pos_encoding=True)
        patches = output.last_hidden_state[:, 1:]  # exclude CLS
    return rearrange(patches, "(b t) n d -> b t n d", b=B)


def parse_gt_state_for_slots(state, num_slots=3):
    """Parse Push-T state into per-object (q, v) for K slots.

    State: [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]

    Returns (B, T, K, 4) where K=3: agent, block, background
    Background gets zeros.
    """
    B, T, _ = state.shape
    dt = 5.0 / 60.0

    per_object = torch.zeros(B, T, num_slots, 4, device=state.device)

    # Object 0: agent — q=(state[0:2]), v=(state[5:7])
    per_object[:, :, 0, :2] = state[:, :, :2]      # agent q
    per_object[:, :, 0, 2:] = state[:, :, 5:7]      # agent v

    # Object 1: block — q=(state[2:4]), v=(finite diff)
    per_object[:, :, 1, :2] = state[:, :, 2:4]      # block q
    # Block velocity via finite difference
    per_object[:, :-1, 1, 2:] = (state[:, 1:, 2:4] - state[:, :-1, 2:4]) / dt

    # Object 2: background — zeros (no physics)

    return per_object


def train(
    checkpoint: str,
    data_dir: str = "/runpod-volume",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda",
    num_workers: int = 4,
):
    print("=== DEMIURGE v0.3 on Push-T ===")
    print()

    # 1. Load frozen LeWM
    print("Loading LeWM encoder...")
    lewm = torch.load(checkpoint, map_location=device, weights_only=False)
    lewm.eval()
    for p in lewm.parameters():
        p.requires_grad = False
    print(f"  LeWM: {sum(p.numel() for p in lewm.parameters()):,} params (frozen)")

    # 2. Load dataset
    print("Loading Push-T dataset...")
    import stable_worldmodel as swm
    from utils import get_img_preprocessor, get_column_normalizer
    import stable_pretraining as spt

    ds = swm.data.HDF5Dataset(
        name="pusht_expert_train",
        num_steps=2,  # just pairs of consecutive frames
        frameskip=5,
        keys_to_load=["pixels", "action", "proprio", "state"],
        keys_to_cache=["action", "proprio", "state"],
        cache_dir=data_dir,
    )

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
    state_normalizer = get_column_normalizer(ds, "state", "state")
    transforms.append(state_normalizer)
    for col in ["action", "proprio"]:
        transforms.append(get_column_normalizer(ds, col, col))
    ds.transform = spt.data.transforms.Compose(*transforms)

    print(f"  Samples: {len(ds)}")

    n_train = int(0.9 * len(ds))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, range(n_train)),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, range(n_train, len(ds))),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # 3. Build v0.3 model
    from v3.demiurge_v3 import DemiurgeV3

    model = DemiurgeV3(
        input_dim=192,
        slot_dim=128,
        static_dim=64,
        state_dim=4,
        num_slots=3,
        dt=5.0 / 60.0,
        lambda_contrast=0.1,
        lambda_energy=0.1,
        lambda_newton=0.1,
        lambda_state=1.0,
    ).to(device)

    counts = model.count_params()
    print(f"  v0.3 params:")
    for k, v in counts.items():
        print(f"    {k}: {v:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Train
    print("Training...")
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {k: [] for k in ["total", "slot", "contrast", "energy", "newton3", "state_t", "state_t1"]}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            pixels = batch["pixels"].to(device)    # (B, 2, 3, 224, 224)
            state = batch["state"].to(device)       # (B, 2, 7)

            # Extract patch tokens for both frames
            patches = extract_patches(lewm, pixels)  # (B, 2, 256, 192)
            patch_t = patches[:, 0]                  # (B, 256, 192)
            patch_t1 = patches[:, 1]                 # (B, 256, 192)

            # Parse GT state into per-object format
            gt_per_object = parse_gt_state_for_slots(state, num_slots=3)  # (B, 2, 3, 4)
            gt_t = gt_per_object[:, 0]    # (B, 3, 4)
            gt_t1 = gt_per_object[:, 1]   # (B, 3, 4)

            # Forward
            out = model(
                patch_tokens_t=patch_t,
                patch_tokens_t1=patch_t1,
                gt_state_t=gt_t,
                gt_state_t1=gt_t1,
            )

            # Backward
            optimizer.zero_grad()
            out["losses"]["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in epoch_losses:
                if k in out["losses"]:
                    val = out["losses"][k]
                    epoch_losses[k].append(val.item() if hasattr(val, "item") else val)

            if batch_idx >= 300:  # cap per epoch
                break

        scheduler.step()
        elapsed = time.time() - t0

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_slot_losses = []
            val_state_losses = []

            with torch.no_grad():
                for vi, vbatch in enumerate(val_loader):
                    vpix = vbatch["pixels"].to(device)
                    vstate = vbatch["state"].to(device)

                    vpatches = extract_patches(lewm, vpix)
                    vgt = parse_gt_state_for_slots(vstate, 3)

                    vout = model(
                        vpatches[:, 0], vpatches[:, 1],
                        gt_state_t=vgt[:, 0], gt_state_t1=vgt[:, 1],
                    )

                    val_slot_losses.append(vout["losses"]["slot"].item())
                    if "state_t1" in vout["losses"]:
                        val_state_losses.append(vout["losses"]["state_t1"].item())

                    if vi >= 100:
                        break

            val_slot = np.mean(val_slot_losses)
            val_state = np.mean(val_state_losses) if val_state_losses else 0
            train_slot = np.mean(epoch_losses["slot"])
            train_energy = np.mean(epoch_losses["energy"]) if epoch_losses["energy"] else 0
            train_newton = np.mean(epoch_losses["newton3"]) if epoch_losses["newton3"] else 0

            marker = " *" if val_slot < best_val else ""
            if val_slot < best_val:
                best_val = val_slot
                torch.save(model.state_dict(), os.path.join(data_dir, "demiurge_v3_best.pt"))

            print(
                f"  Epoch {epoch+1:3d} | "
                f"slot={train_slot:.4f} energy={train_energy:.4f} newton={train_newton:.4f} | "
                f"val_slot={val_slot:.4f} val_state={val_state:.4f} | "
                f"{elapsed:.0f}s{marker}"
            )

    print()
    print(f"Best val slot loss: {best_val:.4f}")

    # 5. Baseline comparisons
    print()
    print("=== BASELINES ===")

    model.eval()

    # Baseline 1: Copy-current slots
    copy_losses = []
    with torch.no_grad():
        for vi, vbatch in enumerate(val_loader):
            vpix = vbatch["pixels"].to(device)
            vpatches = extract_patches(lewm, vpix)

            slots_t, _ = model.slot_attention(vpatches[:, 0])
            slots_t1, _ = model.slot_attention(vpatches[:, 1])

            copy_loss = F.mse_loss(slots_t, slots_t1)
            copy_losses.append(copy_loss.item())
            if vi >= 100:
                break

    copy_baseline = np.mean(copy_losses)
    print(f"  Copy-current slots: {copy_baseline:.4f}")
    print(f"  DEMIURGE v0.3:      {best_val:.4f}")
    print(f"  Improvement:        {(copy_baseline - best_val) / copy_baseline * 100:.1f}%")

    # Baseline 2: State estimation quality
    print()
    print("=== STATE ESTIMATION QUALITY ===")
    from scipy.stats import pearsonr

    est_states, gt_states = [], []
    with torch.no_grad():
        for vi, vbatch in enumerate(val_loader):
            vpix = vbatch["pixels"].to(device)
            vstate = vbatch["state"].to(device)
            vpatches = extract_patches(lewm, vpix)
            vgt = parse_gt_state_for_slots(vstate, 3)

            current = model.extract_slots(vpatches[:, 0])
            est_states.append(current["state"][:, :2].cpu())  # agent + block only
            gt_states.append(vgt[:, 0, :2].cpu())
            if vi >= 100:
                break

    est = torch.cat(est_states).reshape(-1, 4).numpy()
    gt = torch.cat(gt_states).reshape(-1, 4).numpy()

    for i, name in enumerate(["q_x", "q_y", "v_x", "v_y"]):
        r, _ = pearsonr(est[:, i], gt[:, i])
        print(f"  {name}: r={r:.3f}")

    print()
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/runpod-volume/pusht/lejepa_object.ckpt")
    parser.add_argument("--data_dir", default="/runpod-volume")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

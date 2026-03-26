"""
Train DEMIURGE v0.3 with Predict-Update Cycle

Uses forward_temporal() — the Kalman-style loop where slot attention
at frame t+1 is initialized from the IN's prediction at frame t.

This is the architecture as designed: predict, observe, correct, repeat.
Innovation = corrected - predicted = the natural surprise signal.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines", "lewm"))

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from scipy.stats import pearsonr, mannwhitneyu
import time
import argparse

from v3.demiurge_v3 import DemiurgeV3
from v3.slots import hungarian_match, apply_permutation


def extract_patches(lewm, pixels):
    B, T = pixels.shape[:2]
    flat = rearrange(pixels, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        output = lewm.encoder(flat, interpolate_pos_encoding=True)
        patches = output.last_hidden_state[:, 1:]
    return rearrange(patches, "(b t) n d -> b t n d", b=B)


def parse_gt_state(state):
    """Push-T → (B, T, 2, 4): agent + block, each (q_x, q_y, v_x, v_y)."""
    B, T, _ = state.shape
    dt = 5.0 / 60.0
    gt = torch.zeros(B, T, 2, 4, device=state.device)
    gt[:, :, 0, :2] = state[:, :, :2]
    gt[:, :, 0, 2:] = state[:, :, 5:7]
    gt[:, :, 1, :2] = state[:, :, 2:4]
    gt[:, :-1, 1, 2:] = (state[:, 1:, 2:4] - state[:, :-1, 2:4]) / dt
    return gt


def parse_action(action_raw):
    if action_raw.shape[-1] == 10:
        return action_raw.reshape(*action_raw.shape[:-1], 5, 2).mean(dim=-2)
    return action_raw[..., :2]


def train(
    checkpoint: str,
    data_dir: str,
    seq_len: int = 6,
    epochs: int = 60,
    batch_size: int = 20,
    lr: float = 3e-4,
    device: str = "cuda",
    num_workers: int = 4,
):
    print("=== DEMIURGE v0.3 — Temporal Predict-Update Training ===")
    print()

    # Load LeWM
    print("Loading LeWM...")
    lewm = torch.load(checkpoint, map_location=device, weights_only=False)
    lewm.eval()
    for p in lewm.parameters():
        p.requires_grad = False

    # Load dataset
    print("Loading dataset...")
    import stable_worldmodel as swm
    from utils import get_img_preprocessor, get_column_normalizer
    import stable_pretraining as spt

    ds = swm.data.HDF5Dataset(
        name="pusht_expert_train", num_steps=seq_len, frameskip=5,
        keys_to_load=["pixels", "action", "proprio", "state"],
        keys_to_cache=["action", "proprio", "state"],
        cache_dir=data_dir,
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
    for col in ["action", "proprio", "state"]:
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

    # Build model
    model = DemiurgeV3(
        input_dim=192, slot_dim=128, static_dim=64,
        state_dim=4, action_dim=2, num_slots=3, agent_slot=0,
        dt=5.0 / 60.0,
        lambda_energy=0.1, lambda_newton=0.1, lambda_state=1.0,
    ).to(device)
    print(f"  Params: {model.count_params()}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    print("Training with predict-update cycle...")
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        ep = {"total": [], "state": [], "innovation": [], "roundtrip": []}
        t0 = time.time()

        for bi, batch in enumerate(train_loader):
            pixels = batch["pixels"].to(device)
            state = batch["state"].to(device)
            action_raw = batch["action"].to(device)

            patches = extract_patches(lewm, pixels)
            gt = parse_gt_state(state)
            actions = parse_action(action_raw)

            # Temporal predict-update loop
            out = model.forward_temporal(patches, actions=actions, gt_states=gt)

            optimizer.zero_grad()
            out["losses"]["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in ep:
                if k in out["losses"]:
                    ep[k].append(out["losses"][k].item())

            if bi >= 300:
                break

        scheduler.step()
        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_state, val_innov = [], []

            with torch.no_grad():
                for vi, vb in enumerate(val_loader):
                    vpix = vb["pixels"].to(device)
                    vst = vb["state"].to(device)
                    vact = parse_action(vb["action"].to(device))
                    vpatch = extract_patches(lewm, vpix)
                    vgt = parse_gt_state(vst)

                    vout = model.forward_temporal(vpatch, actions=vact, gt_states=vgt)
                    val_state.append(vout["losses"]["state"].item())
                    val_innov.append(vout["losses"]["innovation"].item())

                    if vi >= 100:
                        break

            vs = np.mean(val_state)
            vi = np.mean(val_innov)
            marker = " *" if vs < best_val else ""
            if vs < best_val:
                best_val = vs
                torch.save(model.state_dict(), os.path.join(data_dir, "temporal_best.pt"))

            print(
                f"  Epoch {epoch+1:3d} | "
                f"state={np.mean(ep['state']):.4f} innov={np.mean(ep['innovation']):.4f} "
                f"rt={np.mean(ep['roundtrip']):.4f} | "
                f"val_state={vs:.4f} val_innov={vi:.4f} | "
                f"{elapsed:.0f}s{marker}"
            )

    print(f"\nBest val state: {best_val:.4f}")

    # Load best and evaluate
    model.load_state_dict(torch.load(os.path.join(data_dir, "temporal_best.pt"), weights_only=True))
    model.eval()

    # === VoE with innovation signal ===
    print()
    print("=== VoE: Innovation-Based Surprise ===")
    normal_innov, shifted_innov = [], []

    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            vpix = vb["pixels"].to(device)
            vact = parse_action(vb["action"].to(device))
            vpatch = extract_patches(lewm, vpix)

            # Normal: correct action
            voe_n, _ = model.compute_voe(vpatch[:, 0], vpatch[:, 1], action=vact[:, 0])
            normal_innov.extend(voe_n.cpu().tolist())

            # Shifted: wrong action
            wrong = vact[:, 0] * 3.0 + torch.randn_like(vact[:, 0]) * 0.5
            voe_s, _ = model.compute_voe(vpatch[:, 0], vpatch[:, 1], action=wrong)
            shifted_innov.extend(voe_s.cpu().tolist())

            if vi >= 200:
                break

    nm, sm = np.mean(normal_innov), np.mean(shifted_innov)
    stat, pval = mannwhitneyu(shifted_innov, normal_innov, alternative="greater")
    print(f"  Normal:  {nm:.4f} ± {np.std(normal_innov):.4f}")
    print(f"  Shifted: {sm:.4f} ± {np.std(shifted_innov):.4f}")
    print(f"  Ratio:   {sm/nm:.2f}x")
    print(f"  p-value: {pval:.2e}")

    # === State estimation with temporal tracking ===
    print()
    print("=== State Estimation (Temporal) ===")
    est_all, gt_all = [], []
    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            vpix = vb["pixels"].to(device)
            vst = vb["state"].to(device)
            vpatch = extract_patches(lewm, vpix)
            vgt = parse_gt_state(vst)

            # Run temporal loop — use corrected states from predict-update
            vout = model.forward_temporal(vpatch, actions=parse_action(vb["action"].to(device)), gt_states=vgt)

            # Take last corrected state
            last_state = vout["states"][-1]
            perm = hungarian_match(last_state[:, :, :2], vgt[:, -1, :, :2])
            matched = apply_permutation(last_state, perm)
            est_all.append(matched[:, :2].cpu())
            gt_all.append(vgt[:, -1].cpu())

            if vi >= 200:
                break

    est = torch.cat(est_all).reshape(-1, 4).numpy()
    gt = torch.cat(gt_all).reshape(-1, 4).numpy()
    for i, name in enumerate(["q_x", "q_y", "v_x", "v_y"]):
        r, _ = pearsonr(est[:, i], gt[:, i])
        print(f"  {name}: r={r:.3f}")

    # === Innovation trajectory ===
    print()
    print("=== Innovation Over Sequence ===")
    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            vpix = vb["pixels"].to(device)
            vpatch = extract_patches(lewm, vpix)
            vgt = parse_gt_state(vb["state"].to(device))
            vact = parse_action(vb["action"].to(device))
            vout = model.forward_temporal(vpatch, actions=vact, gt_states=vgt)

            print("  Step  | Innovation (mean)")
            print("  " + "-" * 30)
            for t, inn in enumerate(vout["innovations"]):
                print(f"  t+{t+1:<4} | {inn.mean().item():.4f}")
            break

    print()
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/runpod-volume/checkpoints/lejepa_object.ckpt")
    parser.add_argument("--data_dir", default="/runpod-volume/data")
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(**vars(args))

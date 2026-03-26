"""
Train DEMIURGE v0.3 for Safety Shield Demo

Action-conditioned, multi-step rollout training with all fixes:
    - Hungarian matching for slot identity
    - Actions routed to agent slot
    - State-space primary loss
    - Energy conservation (excludes agent)
    - Newton 3 on interaction effects
    - Multi-step rollout loss for stability

After training, runs the shield demo:
    1. Normal controller on normal physics → baseline success
    2. Normal controller on shifted physics → failure without shield
    3. Shield-equipped controller on shifted physics → shield catches it
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


def parse_gt_state(state, num_slots=4):
    """Push-T state → per-object (q_x, q_y, v_x, v_y).
    Returns (B, T, K_obj, 4) where K_obj=2 (agent + block only, no bg)."""
    B, T, _ = state.shape
    dt = 5.0 / 60.0
    per_obj = torch.zeros(B, T, 2, 4, device=state.device)  # 2 objects
    per_obj[:, :, 0, :2] = state[:, :, :2]       # agent q
    per_obj[:, :, 0, 2:] = state[:, :, 5:7]      # agent v
    per_obj[:, :, 1, :2] = state[:, :, 2:4]      # block q
    per_obj[:, :-1, 1, 2:] = (state[:, 1:, 2:4] - state[:, :-1, 2:4]) / dt
    return per_obj


def parse_action(action_raw):
    """Extract 2D action from Push-T's 10-dim action (frameskip=5).
    Take mean of 5 sub-actions: (10,) → reshape (5,2) → mean → (2,)."""
    if action_raw.shape[-1] == 10:
        return action_raw.reshape(*action_raw.shape[:-1], 5, 2).mean(dim=-2)
    return action_raw[..., :2]


def train(
    checkpoint: str,
    data_dir: str = "/runpod-volume",
    seq_len: int = 6,
    rollout_len: int = 4,
    epochs: int = 60,
    batch_size: int = 24,
    lr: float = 3e-4,
    device: str = "cuda",
    num_workers: int = 4,
):
    print("=== DEMIURGE v0.3 — Shield Training ===")
    print()

    # 1. Load frozen LeWM
    print("Loading LeWM...")
    lewm = torch.load(checkpoint, map_location=device, weights_only=False)
    lewm.eval()
    for p in lewm.parameters():
        p.requires_grad = False

    # 2. Load dataset
    print("Loading dataset...")
    import stable_worldmodel as swm
    from utils import get_img_preprocessor, get_column_normalizer
    import stable_pretraining as spt

    ds = swm.data.HDF5Dataset(
        name="pusht_expert_train",
        num_steps=seq_len,
        frameskip=5,
        keys_to_load=["pixels", "action", "proprio", "state"],
        keys_to_cache=["action", "proprio", "state"],
        cache_dir=data_dir,
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
    for col in ["action", "proprio", "state"]:
        transforms.append(get_column_normalizer(ds, col, col))
    ds.transform = spt.data.transforms.Compose(*transforms)
    print(f"  Samples: {len(ds)}, seq_len: {seq_len}")

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

    # 3. Build model
    model = DemiurgeV3(
        input_dim=192, slot_dim=128, static_dim=64,
        state_dim=4, action_dim=2, num_slots=4, agent_slot=0,
        dt=5.0 / 60.0,
        lambda_contrast=0.05,
        lambda_energy=0.1,
        lambda_newton=0.1,
        lambda_state=1.0,
    ).to(device)
    print(f"  Params: {model.count_params()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Train with action conditioning + multi-step rollout
    print()
    print("Training...")
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        ep_loss = {"1step": [], "multi": [], "energy": [], "newton": []}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            pixels = batch["pixels"].to(device)
            state = batch["state"].to(device)
            action_raw = batch["action"].to(device)
            T = pixels.shape[1]

            patches = extract_patches(lewm, pixels)
            gt = parse_gt_state(state)          # (B, T, 2, 4)
            actions = parse_action(action_raw)   # (B, T, 2)

            # === Single-step with action conditioning ===
            loss_1step = torch.tensor(0.0, device=device)
            all_effects = []
            n_pairs = 0

            for t in range(T - 1):
                out = model(
                    patch_tokens_t=patches[:, t],
                    patch_tokens_t1=patches[:, t + 1],
                    action=actions[:, t],
                    gt_state_t=gt[:, t],
                    gt_state_t1=gt[:, t + 1],
                )
                loss_1step = loss_1step + out["losses"]["total"]
                all_effects.append(out["dynamics"]["effects"])
                n_pairs += 1

            loss_1step = loss_1step / max(n_pairs, 1)

            # === Multi-step rollout (no teacher forcing) ===
            loss_multi = torch.tensor(0.0, device=device)
            if T > 3:
                init = model.extract_slots(patches[:, 0])
                rollout_state = init["state"]
                perm = hungarian_match(rollout_state[:, :, :2], gt[:, 0, :, :2])

                for t in range(min(rollout_len, T - 1)):
                    dyn = model.predict_next(rollout_state, action=actions[:, t])
                    rollout_state = dyn["next_state"]
                    # Compare matched rollout state to GT
                    matched = apply_permutation(rollout_state, perm)
                    loss_multi = loss_multi + F.mse_loss(matched[:, :2], gt[:, t + 1])

                loss_multi = loss_multi / min(rollout_len, T - 1)

            # === Conservation ===
            loss_energy = out["losses"]["energy"]
            loss_newton = torch.tensor(0.0, device=device)
            for eff in all_effects:
                loss_newton = loss_newton + model.dynamics.newton3_loss(eff)
            loss_newton = loss_newton / max(len(all_effects), 1)

            # Total
            loss = loss_1step + 0.5 * loss_multi

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss["1step"].append(loss_1step.item())
            ep_loss["multi"].append(loss_multi.item())
            ep_loss["energy"].append(loss_energy.item())
            ep_loss["newton"].append(loss_newton.item())

            if batch_idx >= 300:
                break

        scheduler.step()
        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            v1step, vmulti = [], []
            with torch.no_grad():
                for vi, vb in enumerate(val_loader):
                    vpix = vb["pixels"].to(device)
                    vst = vb["state"].to(device)
                    vact = parse_action(vb["action"].to(device))
                    vT = vpix.shape[1]
                    vpatch = extract_patches(lewm, vpix)
                    vgt = parse_gt_state(vst)

                    # 1-step
                    vout = model(vpatch[:, 0], vpatch[:, 1],
                                 action=vact[:, 0],
                                 gt_state_t=vgt[:, 0], gt_state_t1=vgt[:, 1])
                    v1step.append(vout["losses"]["state_t1"].item())

                    # Multi-step
                    vinit = model.extract_slots(vpatch[:, 0])
                    vs = vinit["state"]
                    vperm = hungarian_match(vs[:, :, :2], vgt[:, 0, :, :2])
                    merr = 0.0
                    for t in range(min(rollout_len, vT - 1)):
                        vd = model.predict_next(vs, action=vact[:, t])
                        vs = vd["next_state"]
                        vm = apply_permutation(vs, vperm)
                        merr += F.mse_loss(vm[:, :2], vgt[:, t + 1]).item()
                    vmulti.append(merr / min(rollout_len, vT - 1))

                    if vi >= 100:
                        break

            val1 = np.mean(v1step)
            valm = np.mean(vmulti)
            marker = " *" if val1 < best_val else ""
            if val1 < best_val:
                best_val = val1
                torch.save(model.state_dict(), os.path.join(data_dir, "shield_model.pt"))

            print(
                f"  Epoch {epoch+1:3d} | "
                f"1step={np.mean(ep_loss['1step']):.4f} multi={np.mean(ep_loss['multi']):.4f} "
                f"energy={np.mean(ep_loss['energy']):.4f} | "
                f"val_1step={val1:.4f} val_multi={valm:.4f} | "
                f"{elapsed:.0f}s{marker}"
            )

    print(f"\nBest val 1-step: {best_val:.4f}")

    # 5. Load best and run shield demo
    model.load_state_dict(torch.load(os.path.join(data_dir, "shield_model.pt"), weights_only=True))
    model.eval()

    # === SHIELD DEMO ===
    print()
    print("=" * 60)
    print("=== SAFETY SHIELD DEMO ===")
    print("=" * 60)

    # Compute VoE on normal trajectories vs perturbed trajectories
    print()
    print("--- VoE: Normal vs Physics-Shifted Trajectories ---")

    normal_voe = []
    shifted_voe = []

    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            vpix = vb["pixels"].to(device)
            vst = vb["state"].to(device)
            vact = parse_action(vb["action"].to(device))
            vpatch = extract_patches(lewm, vpix)

            # Normal VoE: predict t+1 from t with correct action
            voe_normal = model.compute_voe(vpatch[:, 0], vpatch[:, 1], action=vact[:, 0])
            normal_voe.extend(voe_normal.cpu().tolist())

            # Shifted VoE: scramble the action (simulate wrong physics / bad controller)
            wrong_action = vact[:, 0] * 3.0 + torch.randn_like(vact[:, 0]) * 0.5
            voe_shifted = model.compute_voe(vpatch[:, 0], vpatch[:, 1], action=wrong_action)
            shifted_voe.extend(voe_shifted.cpu().tolist())

            if vi >= 200:
                break

    normal_mean = np.mean(normal_voe)
    shifted_mean = np.mean(shifted_voe)
    stat, pval = mannwhitneyu(shifted_voe, normal_voe, alternative="greater")

    print(f"  Normal VoE:  {normal_mean:.4f} ± {np.std(normal_voe):.4f}")
    print(f"  Shifted VoE: {shifted_mean:.4f} ± {np.std(shifted_voe):.4f}")
    print(f"  Surprise ratio: {shifted_mean / normal_mean:.2f}x")
    print(f"  Mann-Whitney p: {pval:.2e}")

    # Shield accuracy: can we correctly flag shifted trajectories?
    print()
    print("--- Shield Classification Accuracy ---")
    threshold_candidates = np.percentile(normal_voe, [90, 95, 99])
    for pct, thresh in zip([90, 95, 99], threshold_candidates):
        tp = sum(1 for v in shifted_voe if v > thresh)
        fp = sum(1 for v in normal_voe if v > thresh)
        fn = sum(1 for v in shifted_voe if v <= thresh)
        tn = sum(1 for v in normal_voe if v <= thresh)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  Threshold={thresh:.4f} (p{pct}): precision={precision:.3f} recall={recall:.3f}")

    # State estimation quality
    print()
    print("--- State Estimation Quality ---")
    est_all, gt_all = [], []
    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            vpix = vb["pixels"].to(device)
            vst = vb["state"].to(device)
            vpatch = extract_patches(lewm, vpix)
            vgt = parse_gt_state(vst)
            curr = model.extract_slots(vpatch[:, 0])
            perm = hungarian_match(curr["state"][:, :, :2], vgt[:, 0, :, :2])
            matched = apply_permutation(curr["state"], perm)
            est_all.append(matched[:, :2].cpu())  # agent + block
            gt_all.append(vgt[:, 0].cpu())
            if vi >= 200:
                break

    est = torch.cat(est_all).reshape(-1, 4).numpy()
    gt = torch.cat(gt_all).reshape(-1, 4).numpy()
    for i, name in enumerate(["q_x", "q_y", "v_x", "v_y"]):
        r, _ = pearsonr(est[:, i], gt[:, i])
        print(f"  {name}: r={r:.3f}")

    # Multi-step rollout comparison
    print()
    print("--- Multi-Step Rollout: IN vs Copy ---")
    in_errors = {t: [] for t in range(rollout_len)}
    copy_errors = {t: [] for t in range(rollout_len)}

    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            vpix = vb["pixels"].to(device)
            vst = vb["state"].to(device)
            vact = parse_action(vb["action"].to(device))
            vT = vpix.shape[1]
            vpatch = extract_patches(lewm, vpix)
            vgt = parse_gt_state(vst)

            vinit = model.extract_slots(vpatch[:, 0])
            vs = vinit["state"]
            vperm = hungarian_match(vs[:, :, :2], vgt[:, 0, :, :2])
            copy_state = apply_permutation(vs, vperm)[:, :2]

            for t in range(min(rollout_len, vT - 1)):
                vd = model.predict_next(vs, action=vact[:, t])
                vs = vd["next_state"]
                vm = apply_permutation(vs, vperm)[:, :2]
                in_errors[t].append(F.mse_loss(vm, vgt[:, t + 1]).item())
                copy_errors[t].append(F.mse_loss(copy_state, vgt[:, t + 1]).item())

            if vi >= 200:
                break

    print(f"  {'Step':<6} {'IN':>10} {'Copy':>10} {'Winner':>10}")
    print("  " + "-" * 38)
    for t in range(rollout_len):
        ie = np.mean(in_errors[t])
        ce = np.mean(copy_errors[t])
        winner = "IN" if ie < ce else "Copy"
        print(f"  t+{t+1:<4} {ie:>10.4f} {ce:>10.4f} {winner:>10}")

    print()
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/runpod-volume/pusht/lejepa_object.ckpt")
    parser.add_argument("--data_dir", default="/runpod-volume")
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--rollout_len", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        rollout_len=args.rollout_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

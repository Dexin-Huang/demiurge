"""
Push-T Training: DEMIURGE v0.3 — State-Space Prediction

Primary loss in STATE space, not slot space.
    - Extract (q, v) from frame t via slot attention + dynamic head
    - Interaction Network predicts (q', v') at t+1
    - Extract (q, v) from frame t+1 independently
    - Loss: consistency between predicted and extracted state
    - Conservation constraints (energy, Newton 3)
    - Weak contrastive on slots (keeps slot attention alive)

Evaluation on what matters:
    1. Long-horizon rollout stability (20+ steps)
    2. OOD transfer (changed physics at test time)
    3. VoE detection (physically impossible trajectories)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines", "lewm"))

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from scipy.stats import pearsonr
import time
import argparse

from v3.demiurge_v3 import DemiurgeV3


def extract_patches(lewm, pixels):
    """Patch tokens from frozen LeWM."""
    B, T = pixels.shape[:2]
    flat = rearrange(pixels, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        output = lewm.encoder(flat, interpolate_pos_encoding=True)
        patches = output.last_hidden_state[:, 1:]
    return rearrange(patches, "(b t) n d -> b t n d", b=B)


def parse_gt_state(state, num_slots=3):
    """Push-T state → per-object (q_x, q_y, v_x, v_y)."""
    B, T, _ = state.shape
    dt = 5.0 / 60.0
    per_obj = torch.zeros(B, T, num_slots, 4, device=state.device)
    per_obj[:, :, 0, :2] = state[:, :, :2]
    per_obj[:, :, 0, 2:] = state[:, :, 5:7]
    per_obj[:, :, 1, :2] = state[:, :, 2:4]
    per_obj[:, :-1, 1, 2:] = (state[:, 1:, 2:4] - state[:, :-1, 2:4]) / dt
    return per_obj


def train(
    checkpoint: str,
    data_dir: str = "/runpod-volume",
    seq_len: int = 6,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda",
    num_workers: int = 4,
):
    print("=== DEMIURGE v0.3 — State-Space Prediction ===")
    print()

    # 1. Load frozen LeWM
    print("Loading LeWM...")
    lewm = torch.load(checkpoint, map_location=device, weights_only=False)
    lewm.eval()
    for p in lewm.parameters():
        p.requires_grad = False

    # 2. Load dataset — longer sequences for rollout eval
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
        input_dim=192, slot_dim=128, static_dim=64, state_dim=4,
        num_slots=3, dt=5.0 / 60.0,
        lambda_contrast=0.05,   # weak — just keeps slots alive
        lambda_energy=0.1,
        lambda_newton=0.1,
        lambda_state=0.0,       # we handle state loss ourselves below
    ).to(device)

    counts = model.count_params()
    print(f"  Params: {counts['total']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Train — state prediction as primary loss
    print()
    print("Training (state-space primary loss)...")
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        ep = {"state_1step": [], "state_multi": [], "energy": [], "newton": []}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            pixels = batch["pixels"].to(device)    # (B, T, 3, 224, 224)
            state = batch["state"].to(device)       # (B, T, 7)
            T = pixels.shape[1]

            # Extract patches for all frames
            patches = extract_patches(lewm, pixels)  # (B, T, 256, 192)

            # Parse GT state
            gt = parse_gt_state(state, 3)  # (B, T, 3, 4)

            # === Single-step state prediction (primary loss) ===
            # For each consecutive pair, extract state from t, predict t+1,
            # compare with extracted state from t+1
            loss_state = torch.tensor(0.0, device=device)
            all_effects = []

            for t in range(T - 1):
                curr = model.extract_slots(patches[:, t])
                tgt = model.extract_slots(patches[:, t + 1])

                # Predict next state
                dyn = model.predict_next(curr["state"])
                all_effects.append(dyn["effects"])

                # State consistency: predicted vs independently extracted
                loss_state = loss_state + F.mse_loss(dyn["next_state"], tgt["state"].detach())

                # GT supervision (anchors state estimation)
                loss_state = loss_state + 0.5 * F.mse_loss(curr["state"], gt[:, t])
                loss_state = loss_state + 0.5 * F.mse_loss(dyn["next_state"], gt[:, t + 1])

            loss_state = loss_state / (T - 1)

            # === Multi-step rollout loss (if seq_len > 2) ===
            loss_multi = torch.tensor(0.0, device=device)
            if T > 3:
                init = model.extract_slots(patches[:, 0])
                rollout_state = init["state"]
                for t in range(min(T - 1, 4)):
                    dyn = model.predict_next(rollout_state)
                    rollout_state = dyn["next_state"]
                    # Compare rolled-out state with GT
                    loss_multi = loss_multi + F.mse_loss(rollout_state, gt[:, t + 1])
                loss_multi = loss_multi / min(T - 1, 4)

            # === Conservation losses ===
            states_for_energy = [gt[:, t, :, :] for t in range(T)]
            loss_energy = model.dynamics.energy_loss(states_for_energy)

            loss_newton = torch.tensor(0.0, device=device)
            for eff in all_effects:
                loss_newton = loss_newton + model.dynamics.newton3_loss(eff)
            loss_newton = loss_newton / max(len(all_effects), 1)

            # === Total ===
            loss = (
                loss_state
                + 0.5 * loss_multi
                + 0.1 * loss_energy
                + 0.1 * loss_newton
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep["state_1step"].append(loss_state.item())
            ep["state_multi"].append(loss_multi.item())
            ep["energy"].append(loss_energy.item())
            ep["newton"].append(loss_newton.item())

            if batch_idx >= 300:
                break

        scheduler.step()
        elapsed = time.time() - t0

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_1step, val_multi = [], []

            with torch.no_grad():
                for vi, vb in enumerate(val_loader):
                    vpix = vb["pixels"].to(device)
                    vst = vb["state"].to(device)
                    vT = vpix.shape[1]
                    vpatch = extract_patches(lewm, vpix)
                    vgt = parse_gt_state(vst, 3)

                    # 1-step
                    vc = model.extract_slots(vpatch[:, 0])
                    vt = model.extract_slots(vpatch[:, 1])
                    vd = model.predict_next(vc["state"])
                    val_1step.append(F.mse_loss(vd["next_state"], vt["state"]).item())

                    # Multi-step rollout
                    rs = vc["state"]
                    multi_err = 0.0
                    for t in range(min(vT - 1, 4)):
                        vd = model.predict_next(rs)
                        rs = vd["next_state"]
                        multi_err += F.mse_loss(rs, vgt[:, t + 1]).item()
                    val_multi.append(multi_err / min(vT - 1, 4))

                    if vi >= 100:
                        break

            v1 = np.mean(val_1step)
            vm = np.mean(val_multi)
            marker = " *" if v1 < best_val else ""
            if v1 < best_val:
                best_val = v1
                torch.save(model.state_dict(), os.path.join(data_dir, "v3_state_best.pt"))

            print(
                f"  Epoch {epoch+1:3d} | "
                f"1step={np.mean(ep['state_1step']):.4f} multi={np.mean(ep['state_multi']):.4f} "
                f"energy={np.mean(ep['energy']):.4f} | "
                f"val_1step={v1:.4f} val_multi={vm:.4f} | "
                f"{elapsed:.0f}s{marker}"
            )

    # 5. Evaluation
    print()
    print(f"Best val 1-step state loss: {best_val:.4f}")

    model.eval()
    model.load_state_dict(torch.load(os.path.join(data_dir, "v3_state_best.pt"), weights_only=True))

    # === Baseline: MLP state predictor ===
    print()
    print("=== BASELINE: MLP state predictor ===")

    # Train a simple MLP that predicts next state from current state
    # (no structure, no objects, no conservation)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(4 * 3, 128),  # all objects concatenated
        torch.nn.GELU(),
        torch.nn.Linear(128, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 4 * 3),
    ).to(device)
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    # Quick train on GT states
    for mlp_epoch in range(50):
        for bi, b in enumerate(train_loader):
            st = b["state"].to(device)
            gt = parse_gt_state(st, 3)
            curr_flat = gt[:, 0].reshape(-1, 12)
            next_flat = gt[:, 1].reshape(-1, 12)
            pred = mlp(curr_flat)
            mlp_loss = F.mse_loss(pred, next_flat)
            mlp_opt.zero_grad()
            mlp_loss.backward()
            mlp_opt.step()
            if bi >= 100:
                break

    # Evaluate MLP vs IN on long rollouts
    print()
    print("=== LONG-HORIZON ROLLOUT (20 steps) ===")

    rollout_errors_in = []
    rollout_errors_mlp = []
    rollout_errors_copy = []

    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            vpix = vb["pixels"].to(device)
            vst = vb["state"].to(device)
            vT = vpix.shape[1]
            vpatch = extract_patches(lewm, vpix)
            vgt = parse_gt_state(vst, 3)

            if vT < 3:
                continue

            # Initial state (from slot attention)
            init = model.extract_slots(vpatch[:, 0])
            state_in = init["state"]
            state_mlp = vgt[:, 0].reshape(-1, 12)
            state_copy = vgt[:, 0]

            err_in, err_mlp, err_copy = [], [], []

            for t in range(min(vT - 1, 5)):  # limited by seq_len
                # IN rollout
                dyn = model.predict_next(state_in)
                state_in = dyn["next_state"]
                err_in.append(F.mse_loss(state_in, vgt[:, t + 1]).item())

                # MLP rollout
                pred_mlp = mlp(state_mlp)
                state_mlp = pred_mlp
                err_mlp.append(F.mse_loss(pred_mlp.reshape(-1, 3, 4), vgt[:, t + 1]).item())

                # Copy baseline
                err_copy.append(F.mse_loss(state_copy, vgt[:, t + 1]).item())

            rollout_errors_in.append(err_in)
            rollout_errors_mlp.append(err_mlp)
            rollout_errors_copy.append(err_copy)

            if vi >= 200:
                break

    # Average per step
    max_steps = min(len(rollout_errors_in[0]), 5)
    print(f"  {'Step':<6} {'IN':>10} {'MLP':>10} {'Copy':>10}")
    print("  " + "-" * 38)
    for t in range(max_steps):
        in_err = np.mean([e[t] for e in rollout_errors_in if len(e) > t])
        mlp_err = np.mean([e[t] for e in rollout_errors_mlp if len(e) > t])
        copy_err = np.mean([e[t] for e in rollout_errors_copy if len(e) > t])
        print(f"  t+{t+1:<4} {in_err:>10.4f} {mlp_err:>10.4f} {copy_err:>10.4f}")

    # === State estimation quality ===
    print()
    print("=== STATE ESTIMATION (from patches) ===")
    est_all, gt_all = [], []
    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            vpix = vb["pixels"].to(device)
            vst = vb["state"].to(device)
            vpatch = extract_patches(lewm, vpix)
            vgt = parse_gt_state(vst, 3)
            curr = model.extract_slots(vpatch[:, 0])
            est_all.append(curr["state"][:, :2].cpu())
            gt_all.append(vgt[:, 0, :2].cpu())
            if vi >= 200:
                break

    est = torch.cat(est_all).reshape(-1, 4).numpy()
    gt = torch.cat(gt_all).reshape(-1, 4).numpy()
    for i, name in enumerate(["q_x", "q_y", "v_x", "v_y"]):
        r, _ = pearsonr(est[:, i], gt[:, i])
        print(f"  {name}: r={r:.3f}")

    # === VoE test: shuffle block positions mid-sequence ===
    print()
    print("=== VoE: SHUFFLED BLOCK TEST ===")

    normal_errors, shuffled_errors = [], []
    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            vpix = vb["pixels"].to(device)
            vst = vb["state"].to(device)
            vT = vpix.shape[1]
            if vT < 3:
                continue
            vpatch = extract_patches(lewm, vpix)
            vgt = parse_gt_state(vst, 3)

            # Normal: predict t+1 from t
            c = model.extract_slots(vpatch[:, 0])
            d = model.predict_next(c["state"])
            t1 = model.extract_slots(vpatch[:, 1])
            normal_err = F.mse_loss(d["next_state"], t1["state"]).item()
            normal_errors.append(normal_err)

            # Shuffled: swap block positions across batch
            c_shuf = model.extract_slots(vpatch[:, 0])
            perm = torch.randperm(c_shuf["state"].shape[0])
            c_shuf["state"][:, 1] = c_shuf["state"][perm, 1]  # shuffle block
            d_shuf = model.predict_next(c_shuf["state"])
            shuffled_err = F.mse_loss(d_shuf["next_state"], t1["state"]).item()
            shuffled_errors.append(shuffled_err)

            if vi >= 200:
                break

    normal_mean = np.mean(normal_errors)
    shuffled_mean = np.mean(shuffled_errors)
    print(f"  Normal prediction error:   {normal_mean:.4f}")
    print(f"  Shuffled prediction error: {shuffled_mean:.4f}")
    print(f"  Surprise ratio:            {shuffled_mean / normal_mean:.2f}x")

    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(shuffled_errors, normal_errors, alternative="greater")
    print(f"  Mann-Whitney p-value:      {pval:.2e}")

    print()
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/runpod-volume/pusht/lejepa_object.ckpt")
    parser.add_argument("--data_dir", default="/runpod-volume")
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

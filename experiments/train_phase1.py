"""
Phase 1: Train Hybrid Belief Simulator on Ground-Truth 2D Physics

Bypasses the state estimator — feeds GT (q, v) directly into the simulator.
Trains via observation loss: can the simulator + observer predict future
LeWM-style embeddings from simulated state?

Since we don't have LeWM embeddings for pymunk data, we use a simpler
proxy: predict future (q, v) directly. This tests whether the simulator
learns correct dynamics before we hook it up to the perceptual backbone.

Validation checks:
  1. Does trajectory prediction improve over training?
  2. Do material codes z correlate with actual mass/friction?
  3. Do interaction modes switch at contact events?
  4. Is energy monotonically decreasing (passivity)?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr

from simulator.hybrid_simulator import HybridBeliefSimulator
from simulator.material_belief import MaterialBelief
from estimator.object_tracker import ObjectTracker
from experiments.sim2d import generate_dataset


def train_phase1(
    n_trajectories: int = 2000,
    n_steps: int = 60,
    n_objects: int = 3,
    rollout_len: int = 8,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
):
    print("=== PHASE 1: Hybrid Simulator on GT Physics ===")
    print()

    # 1. Generate data
    print(f"Generating {n_trajectories} trajectories ({n_objects} objects, {n_steps} steps)...")
    data = generate_dataset(
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        n_objects=n_objects,
        seed=seed,
    )

    positions = data["positions"]     # (N, T+1, K, 2)
    velocities = data["velocities"]   # (N, T+1, K, 2)
    masses = data["masses"]           # (N, K)
    frictions = data["frictions"]     # (N, K)
    contacts = data["contact_matrices"]  # (N, T, K, K)

    N, T_plus_1, K, D = positions.shape
    T = T_plus_1 - 1
    print(f"  Positions: {positions.shape}")
    print(f"  Contacts: {int(contacts.sum())} total events")
    print()

    # 2. Build model
    simulator = HybridBeliefSimulator(
        state_dim=D,
        material_dim=8,
        force_hidden=64,
        gravity=(0.0, -300.0 / 500.0),  # normalized
        dt=1.0 / 60.0,
        innovation_threshold=0.3,
    ).to(device)

    tracker = ObjectTracker(contact_threshold=0.06)  # normalized coords

    param_count = sum(p.numel() for p in simulator.parameters() if p.requires_grad)
    print(f"Simulator params: {param_count:,}")

    optimizer = torch.optim.AdamW(simulator.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training loop
    # For each trajectory, pick a random starting timestep and roll forward
    print("Training...")
    print()

    train_N = int(0.8 * N)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        simulator.train()
        epoch_losses = {"dynamics": [], "passivity": [], "material_kl": []}

        # Shuffle training trajectories
        perm = torch.randperm(train_N)
        for batch_start in range(0, train_N, batch_size):
            batch_idx = perm[batch_start:batch_start + batch_size]

            # Random start time (leave room for rollout)
            t0 = np.random.randint(0, T - rollout_len)

            # Get batch
            q = positions[batch_idx, t0].to(device)        # (B, K, 2)
            v = velocities[batch_idx, t0].to(device)       # (B, K, 2)
            B = q.shape[0]

            # Initialize material belief
            z = torch.zeros(B, K, 8, device=device)
            sigma = torch.ones(B, K, 8, device=device)
            contact_prev = torch.zeros(B, K, K, device=device)

            # Roll forward and compare with GT
            pred_qs, pred_vs = [q], [v]
            energies = [simulator.energy(q, v, z)]

            for step in range(rollout_len):
                pairwise = tracker.compute_pairwise(q, v)
                innovation = tracker.detect_mechanical_innovation(
                    contact_prev, pairwise["contact"],
                    (v - pred_vs[-1]) / simulator.dt if step > 0 else torch.zeros_like(v),
                )

                out = simulator.step(q, v, z, sigma, innovation, pairwise)
                q, v, z, sigma = out["q"], out["v"], out["z"], out["sigma"]

                pred_qs.append(q)
                pred_vs.append(v)
                energies.append(simulator.energy(q, v, z))
                contact_prev = pairwise["contact"]

            # Target: GT positions and velocities
            gt_q = positions[batch_idx, t0:t0 + rollout_len + 1].to(device)
            gt_v = velocities[batch_idx, t0:t0 + rollout_len + 1].to(device)

            pred_q = torch.stack(pred_qs, dim=1)  # (B, R+1, K, 2)
            pred_v = torch.stack(pred_vs, dim=1)

            # Losses
            loss_q = F.mse_loss(pred_q, gt_q)
            loss_v = F.mse_loss(pred_v, gt_v)
            loss_dynamics = loss_q + 0.5 * loss_v

            loss_passivity = simulator.passivity_loss(energies)
            loss_kl = simulator.material_belief.kl_loss(z, sigma)

            loss = loss_dynamics + 0.1 * loss_passivity + 0.01 * loss_kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(simulator.parameters(), 1.0)
            optimizer.step()

            epoch_losses["dynamics"].append(loss_dynamics.item())
            epoch_losses["passivity"].append(loss_passivity.item())
            epoch_losses["material_kl"].append(loss_kl.item())

        scheduler.step()

        # Validation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            simulator.eval()
            val_losses = []

            with torch.no_grad():
                for val_start in range(train_N, N, batch_size):
                    val_idx = list(range(val_start, min(val_start + batch_size, N)))
                    q = positions[val_idx, 0].to(device)
                    v = velocities[val_idx, 0].to(device)
                    B_val = q.shape[0]
                    z = torch.zeros(B_val, K, 8, device=device)
                    sigma = torch.ones(B_val, K, 8, device=device)
                    contact_prev = torch.zeros(B_val, K, K, device=device)

                    pred_qs = [q]
                    for step in range(rollout_len):
                        pairwise = tracker.compute_pairwise(q, v)
                        innovation = tracker.detect_mechanical_innovation(
                            contact_prev, pairwise["contact"],
                            torch.zeros_like(v),
                        )
                        out = simulator.step(q, v, z, sigma, innovation, pairwise)
                        q, v, z, sigma = out["q"], out["v"], out["z"], out["sigma"]
                        pred_qs.append(q)
                        contact_prev = pairwise["contact"]

                    pred_q = torch.stack(pred_qs, dim=1)
                    gt_q = positions[val_idx, :rollout_len + 1].to(device)
                    val_losses.append(F.mse_loss(pred_q, gt_q).item())

            val_loss = np.mean(val_losses)
            dyn_loss = np.mean(epoch_losses["dynamics"])
            pass_loss = np.mean(epoch_losses["passivity"])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                marker = " *"
            else:
                marker = ""

            print(f"  Epoch {epoch+1:3d} | train_dyn={dyn_loss:.4f} passivity={pass_loss:.4f} | val={val_loss:.4f}{marker}")

    # 4. Validation: material code probes
    print()
    print("=== MATERIAL CODE ANALYSIS ===")
    simulator.eval()

    all_z, all_mass, all_fric = [], [], []
    with torch.no_grad():
        for i in range(train_N, N):
            q = positions[i, 0:1].to(device)
            v = velocities[i, 0:1].to(device)
            z = torch.zeros(1, K, 8, device=device)
            sigma = torch.ones(1, K, 8, device=device)
            contact_prev = torch.zeros(1, K, K, device=device)

            # Run forward to let material codes accumulate evidence
            for step in range(min(30, T)):
                pairwise = tracker.compute_pairwise(q, v)
                innovation = tracker.detect_mechanical_innovation(
                    contact_prev, pairwise["contact"],
                    torch.zeros_like(v),
                )
                out = simulator.step(q, v, z, sigma, innovation, pairwise)
                q, v, z, sigma = out["q"], out["v"], out["z"], out["sigma"]
                contact_prev = pairwise["contact"]

            all_z.append(z[0].cpu())
            all_mass.append(masses[i])
            all_fric.append(frictions[i])

    all_z = torch.stack(all_z).reshape(-1, 8).numpy()   # (N_val * K, 8)
    all_mass = torch.stack(all_mass).reshape(-1).numpy()  # (N_val * K,)
    all_fric = torch.stack(all_fric).reshape(-1).numpy()

    # Probe: can we linearly predict mass/friction from z?
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    for name, target in [("mass", all_mass), ("friction", all_fric)]:
        reg = LinearRegression()
        scores = cross_val_score(reg, all_z, target, cv=5, scoring="r2")
        r2 = scores.mean()

        # Pearson r
        reg.fit(all_z, target)
        pred = reg.predict(all_z)
        r, _ = pearsonr(pred, target)
        print(f"  {name:>10}: r={r:.3f}, R²={r2:.3f}")

    # 5. Mode analysis
    print()
    print("=== MODE ANALYSIS ===")
    with torch.no_grad():
        # Pick a trajectory with contacts
        for i in range(train_N, N):
            if contacts[i].sum() > 5:
                q = positions[i, 0:1].to(device)
                v = velocities[i, 0:1].to(device)
                z = torch.zeros(1, K, 8, device=device)
                sigma = torch.ones(1, K, 8, device=device)
                contact_prev = torch.zeros(1, K, K, device=device)

                mode_history = []
                gt_contact_history = []

                for step in range(min(30, T)):
                    pairwise = tracker.compute_pairwise(q, v)
                    innovation = tracker.detect_mechanical_innovation(
                        contact_prev, pairwise["contact"],
                        torch.zeros_like(v),
                    )
                    out = simulator.step(q, v, z, sigma, innovation, pairwise)
                    q, v, z, sigma = out["q"], out["v"], out["z"], out["sigma"]

                    mode_history.append(out["modes"][0].cpu())
                    gt_contact_history.append(contacts[i, step] if step < contacts.shape[1] else np.zeros((K, K)))
                    contact_prev = pairwise["contact"]

                # Check if predicted modes correlate with GT contacts
                pred_active = torch.stack(mode_history).numpy() > 0  # any non-NONE mode
                gt_active = np.stack(gt_contact_history) > 0

                agreement = (pred_active == gt_active).mean()
                print(f"  Mode-contact agreement: {agreement:.1%}")
                print(f"  Predicted active modes: {pred_active.sum()}")
                print(f"  GT contact events: {gt_active.sum()}")
                break

    print()
    print("=== PHASE 1 COMPLETE ===")


if __name__ == "__main__":
    train_phase1()

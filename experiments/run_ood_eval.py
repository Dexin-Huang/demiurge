"""
Run OOD Physics Evaluation with Real Trajectories

The correct evaluation: generate actual Push-T trajectories with
modified physics parameters, encode them with LeWM, and measure
whether the predict-update innovation signal detects the change.

No GT state manipulation. Everything through pixels.

Usage:
    python experiments/run_ood_eval.py \
        --checkpoint /runpod-volume/checkpoints/lejepa_object.ckpt \
        --model_weights /runpod-volume/data/temporal_best.pt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines", "lewm"))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu
import argparse

from v3.demiurge_v3 import DemiurgeV3
from eval.suite.pusht_ood import (
    generate_ood_trajectories,
    evaluate_with_model,
    PhysicsShift,
)


def run(
    checkpoint: str,
    model_weights: str,
    n_episodes: int = 30,
    episode_length: int = 30,
    device: str = "cuda",
):
    print("=" * 60)
    print("OOD Physics Evaluation — Real Trajectories")
    print("=" * 60)
    print()

    # Load models
    print("Loading LeWM...")
    lewm = torch.load(checkpoint, map_location=device, weights_only=False)
    lewm.eval()
    for p in lewm.parameters():
        p.requires_grad = False

    print("Loading DEMIURGE v0.3...")
    model = DemiurgeV3(
        input_dim=192, slot_dim=128, static_dim=64,
        state_dim=4, action_dim=2, num_slots=4, agent_slot=0,
        dt=5.0 / 60.0,
    ).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.eval()

    # Generate OOD trajectories
    print()
    print("Generating OOD trajectories...")
    shifts = [
        PhysicsShift.NORMAL,
        PhysicsShift.FRICTION_HIGH,
        PhysicsShift.FRICTION_LOW,
        PhysicsShift.BLOCK_HEAVY,
        PhysicsShift.BLOCK_LIGHT,
        PhysicsShift.DAMPING_HIGH,
    ]

    trajectories = generate_ood_trajectories(
        n_episodes=n_episodes,
        episode_length=episode_length,
        shifts=shifts,
        seed=42,
    )

    # Evaluate
    print()
    print("Evaluating innovation signals...")
    results = evaluate_with_model(model, lewm, trajectories, device=device)

    # Print results
    print()
    print("=" * 60)
    print("Results: Innovation-Based Physics Anomaly Detection")
    print("=" * 60)
    print()
    print(f"  Normal baseline:  {results.get('normal_mean', 0):.4f} ± {results.get('normal_std', 0):.4f}")
    print()
    print(f"  {'Shift':<20} {'Mean':>8} {'Ratio':>8} {'AUROC':>8}")
    print("  " + "-" * 48)

    for shift in shifts:
        if shift == PhysicsShift.NORMAL:
            continue
        name = shift.name
        mean = results.get(f"{name}_mean", 0)
        ratio = results.get(f"{name}_ratio", 1)
        auroc = results.get(f"{name}_auroc", 0.5)
        print(f"  {name:<20} {mean:>8.4f} {ratio:>7.2f}x {auroc:>8.3f}")

    # Overall AUROC: all physics shifts vs normal
    print()
    normal_scores = []
    physics_scores = []
    for shift_name, scores in results.items():
        pass  # results is flat dict, not scores

    print("=" * 60)
    print()
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/runpod-volume/checkpoints/lejepa_object.ckpt")
    parser.add_argument("--model_weights", default="/runpod-volume/data/temporal_best.pt")
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--episode_length", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run(**vars(args))

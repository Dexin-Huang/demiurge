"""
Run Full Evaluation Suite

One command to reproduce the complete results table:
    python experiments/run_eval.py --checkpoint <path> --data_dir <path>

Outputs:
    - AUROC / AUPRC across 5 seeds
    - Per-perturbation breakdown
    - False positive rate on nuisance controls
    - Median detection delay
    - Baseline comparison (JEPA surprise vs DEMIURGE innovation)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines", "lewm"))

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import argparse

from v3.demiurge_v3 import DemiurgeV3
# TODO: EvalHarness was removed (flawed perturbation approach).
# Replace with pusht_ood-based evaluation when updating this script.
# from eval.suite.harness import EvalHarness


def extract_patches(lewm, pixels):
    B, T = pixels.shape[:2]
    flat = rearrange(pixels, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        output = lewm.encoder(flat, interpolate_pos_encoding=True)
        patches = output.last_hidden_state[:, 1:]
    return rearrange(patches, "(b t) n d -> b t n d", b=B)


def parse_gt_state(state):
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


def jepa_surprise_baseline(lewm, pixels):
    """Baseline: JEPA CLS embedding prediction error (no structure)."""
    B, T = pixels.shape[:2]
    flat = rearrange(pixels, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        output = lewm.encoder(flat, interpolate_pos_encoding=True)
        cls = lewm.projector(output.last_hidden_state[:, 0])
    cls = rearrange(cls, "(b t) d -> b t d", b=B)

    # Surprise = MSE between consecutive CLS embeddings
    surprises = []
    for t in range(T - 1):
        surprise = (cls[:, t + 1] - cls[:, t]).pow(2).sum(dim=-1)
        surprises.append(surprise)
    return surprises  # list of (B,) tensors


def run(
    checkpoint: str,
    model_weights: str,
    data_dir: str,
    seq_len: int = 6,
    batch_size: int = 20,
    n_seeds: int = 5,
    device: str = "cuda",
    num_workers: int = 4,
):
    print("=" * 60)
    print("DEMIURGE v0.3 — Full Evaluation Suite")
    print("=" * 60)
    print()

    # Load LeWM
    print("Loading LeWM...")
    lewm = torch.load(checkpoint, map_location=device, weights_only=False)
    lewm.eval()
    for p in lewm.parameters():
        p.requires_grad = False

    # Load model
    print("Loading DEMIURGE v0.3...")
    model = DemiurgeV3(
        input_dim=192, slot_dim=128, static_dim=64,
        state_dim=4, action_dim=2, num_slots=4, agent_slot=0,
        dt=5.0 / 60.0,
    ).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.eval()
    print(f"  Params: {model.count_params()}")

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

    # Test split (last 10%)
    n_test = int(0.1 * len(ds))
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, range(len(ds) - n_test, len(ds))),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"  Test samples: {n_test}")
    print()

    # Run DEMIURGE evaluation
    print("Running DEMIURGE evaluation...")
    harness = EvalHarness(
        model=model,
        extract_patches_fn=lambda l, p: extract_patches(l, p),
        parse_gt_state_fn=parse_gt_state,
        parse_action_fn=parse_action,
        threshold_percentile=95,
    )
    results = harness.run_multi_seed(
        test_loader, lewm, n_seeds=n_seeds, device=device,
    )
    harness.print_table(results)

    # Run JEPA baseline for comparison
    print()
    print("Running JEPA surprise baseline...")
    jepa_scores_normal = []
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            pixels = batch["pixels"].to(device)
            surprises = jepa_surprise_baseline(lewm, pixels)
            for s in surprises:
                jepa_scores_normal.extend(s.cpu().tolist())
            if bi >= 200:
                break

    print(f"  JEPA surprise (normal): {np.mean(jepa_scores_normal):.4f} ± {np.std(jepa_scores_normal):.4f}")
    print()

    print("=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/runpod-volume/checkpoints/lejepa_object.ckpt")
    parser.add_argument("--model_weights", default="/runpod-volume/data/temporal_best.pt")
    parser.add_argument("--data_dir", default="/runpod-volume/data")
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run(**vars(args))

"""
Planning Evaluation: Structured vs LeWM CEM

The experiment: same Push-T environment, same CEM budget.
LeWM plans in latent space. DEMIURGE plans in slot/state space.
Who reaches the goal more reliably?

Test under:
    1. Normal physics (IID) — should be comparable
    2. Shifted physics (OOD) — structured should degrade more gracefully

This is the integration of the world model into RL (model-predictive control).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines", "lewm"))

import torch
import numpy as np
import argparse

from v3.demiurge_v3 import DemiurgeV3
from v3.planner import StructuredPlanner


def run_planning_eval(
    checkpoint: str,
    model_weights: str,
    n_episodes: int = 50,
    horizon: int = 5,
    n_samples: int = 300,
    device: str = "cuda",
):
    print("=" * 60)
    print("Planning Evaluation: Structured vs LeWM CEM")
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
        state_dim=6, action_dim=2, num_slots=4, agent_slot=0,
        dt=5.0 / 60.0,
    ).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.eval()

    # Create planner
    planner = StructuredPlanner(
        model=model, lewm=lewm,
        n_samples=n_samples, horizon=horizon,
        n_iterations=10, top_k=30,
    )

    # Create Push-T environment
    print("Creating Push-T environment...")
    import stable_worldmodel as swm

    world = swm.World(
        "swm/PushT-v1",
        num_envs=1,
        image_shape=[224, 224],
        max_episode_steps=50,
        seed=42,
    )

    # Load dataset for goal states
    from utils import get_img_preprocessor
    preprocess = get_img_preprocessor(source="pixels", target="pixels", img_size=224)

    print()
    print("=" * 60)
    print("Test 1: Normal Physics")
    print("=" * 60)

    structured_successes = 0
    lewm_successes = 0
    rng = np.random.RandomState(42)

    for ep in range(n_episodes):
        obs, info = world.envs.reset(seed=int(rng.randint(0, 100000)))

        # Get current frame
        env = world.envs.envs[0].unwrapped
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frame_t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            frame_t = (frame_t - mean) / std

        # Get initial state
        init_state = env._get_obs()

        # Plan with structured model
        # For now, just test that the planner runs and produces valid actions
        try:
            actions = planner.plan(frame_t, frame_t)  # self-goal for now
            structured_successes += 1
        except Exception as e:
            print(f"  Structured planner failed: {e}")

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}")

    print(f"\n  Structured planner ran successfully: {structured_successes}/{n_episodes}")
    print()
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/runpod-volume/checkpoints/lejepa_object.ckpt")
    parser.add_argument("--model_weights", default="/runpod-volume/data/temporal_best.pt")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_planning_eval(**vars(args))

"""
Planning Evaluation: Structured vs LeWM CEM on Push-T

End-to-end closed-loop evaluation:
    1. Sample start/goal pairs from dataset
    2. Plan with DEMIURGE structured CEM
    3. Plan with LeWM native CEM (via stable-worldmodel API)
    4. Execute plans in Push-T environment
    5. Measure goal achievement (block pose distance)
    6. Repeat under normal AND shifted physics

Uses LeWM's evaluation protocol:
    - horizon=5, action_block=5, receding_horizon=5
    - CEM: 300 samples, 30 iterations, top-30 elites
    - 50 evaluation episodes, seed=42
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


def preprocess_frame(frame, device):
    """Convert raw env render to LeWM input format."""
    if isinstance(frame, np.ndarray):
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    else:
        t = frame.float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(device)


def evaluate_planner(
    planner,
    env,
    start_states,
    goal_states,
    device,
    action_block=5,
    eval_budget=50,
    label="Structured",
):
    """Run closed-loop planning evaluation.

    For each (start, goal) pair:
        1. Reset env to start state
        2. Plan action sequence toward goal
        3. Execute action_block steps
        4. Measure final block pose distance to goal

    Returns list of final distances (lower = better).
    """
    distances = []

    for ep, (start, goal) in enumerate(zip(start_states, goal_states)):
        env_unwrapped = env.envs[0].unwrapped

        # Reset to start state
        env.reset()
        try:
            env_unwrapped._set_state(state=start)
        except Exception:
            continue

        # Get current and goal frames
        current_frame = preprocess_frame(env_unwrapped.render(), device)

        # Set goal state temporarily to render goal
        saved_state = env_unwrapped._get_obs()
        try:
            env_unwrapped._set_state(state=goal)
            goal_frame = preprocess_frame(env_unwrapped.render(), device)
            env_unwrapped._set_state(state=saved_state)
        except Exception:
            goal_frame = current_frame  # fallback

        # Plan
        agent_pos = torch.tensor(start[:2], device=device, dtype=torch.float32)
        try:
            actions = planner.plan(current_frame, goal_frame, agent_pos=agent_pos)
        except Exception as e:
            if (ep + 1) <= 3:
                print(f"  {label} plan failed ep {ep}: {e}")
            distances.append(float('inf'))
            continue

        # Execute action_block steps
        for t in range(min(action_block, actions.shape[0])):
            action = actions[t].cpu().numpy()
            # Push-T expects (num_envs, action_dim)
            action_env = action.reshape(1, -1)
            try:
                env.step(action_env)
            except Exception:
                break

        # Measure final block pose distance to goal
        final_state = env_unwrapped._get_obs()
        block_pos = final_state[2:4]
        goal_block_pos = goal[2:4]
        dist = np.sqrt(((block_pos - goal_block_pos) ** 2).sum())
        distances.append(dist)

        if (ep + 1) % 10 == 0:
            print(f"  {label} ep {ep+1}: mean_dist={np.mean(distances):.2f}")

    return distances


def run(
    checkpoint: str,
    model_weights: str,
    n_episodes: int = 50,
    device: str = "cuda",
):
    print("=" * 60)
    print("Planning Eval: Structured CEM vs Baseline")
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

    planner = StructuredPlanner(
        model=model, lewm=lewm,
        n_samples=300, horizon=5, n_iterations=30, top_k=30,
    )

    # Create Push-T env
    print("Creating environment...")
    import stable_worldmodel as swm

    world = swm.World(
        "swm/PushT-v1", num_envs=1,
        image_shape=[224, 224], max_episode_steps=50, seed=42,
    )

    # Sample start/goal pairs from dataset
    print("Sampling start/goal pairs...")
    ds = swm.data.HDF5Dataset(
        name="pusht_expert_train", num_steps=30, frameskip=5,
        keys_to_load=["state"], keys_to_cache=["state"],
        cache_dir=os.environ.get("STABLEWM_HOME", "/runpod-volume/data"),
    )

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(ds))[:n_episodes]
    start_states, goal_states = [], []
    for idx in indices:
        sample = ds[int(idx)]
        state = sample["state"].numpy()
        start_states.append(state[0])     # first frame
        goal_states.append(state[-1])     # last frame (25 steps later)

    # === Test 1: Normal Physics ===
    print()
    print("=" * 60)
    print("Test 1: Normal Physics")
    print("=" * 60)

    structured_dists = evaluate_planner(
        planner, world, start_states, goal_states, device, label="Structured",
    )

    # Random baseline
    random_dists = []
    for ep in range(n_episodes):
        env = world.envs.envs[0].unwrapped
        world.envs.reset()
        try:
            env._set_state(state=start_states[ep])
        except Exception:
            random_dists.append(float('inf'))
            continue
        for _ in range(5):
            action = rng.uniform(-1, 1, size=(1, 2)).astype(np.float32)
            try:
                world.step(action)
            except Exception:
                break
        final = env._get_obs()
        dist = np.sqrt(((final[2:4] - goal_states[ep][2:4]) ** 2).sum())
        random_dists.append(dist)

    print()
    s_valid = [d for d in structured_dists if d < float('inf')]
    r_valid = [d for d in random_dists if d < float('inf')]
    print(f"  Structured: {np.mean(s_valid):.2f} ± {np.std(s_valid):.2f} (n={len(s_valid)})")
    print(f"  Random:     {np.mean(r_valid):.2f} ± {np.std(r_valid):.2f} (n={len(r_valid)})")

    if np.mean(s_valid) < np.mean(r_valid):
        print("  → Structured planner beats random!")
    else:
        print("  → Structured planner does not beat random yet.")

    # === Test 2: Shifted Physics (heavy block) ===
    print()
    print("=" * 60)
    print("Test 2: Heavy Block (mass 5x)")
    print("=" * 60)

    # Modify physics
    env = world.envs.envs[0].unwrapped
    original_mass = env.block.mass

    def set_heavy():
        env.block.mass = original_mass * 5

    def reset_physics():
        env.block.mass = original_mass

    set_heavy()
    structured_ood = evaluate_planner(
        planner, world, start_states[:20], goal_states[:20], device,
        label="Structured-OOD",
    )

    reset_physics()
    set_heavy()
    random_ood = []
    for ep in range(20):
        world.envs.reset()
        try:
            env._set_state(state=start_states[ep])
        except Exception:
            random_ood.append(float('inf'))
            continue
        for _ in range(5):
            action = rng.uniform(-1, 1, size=(1, 2)).astype(np.float32)
            try:
                world.step(action)
            except Exception:
                break
        final = env._get_obs()
        dist = np.sqrt(((final[2:4] - goal_states[ep][2:4]) ** 2).sum())
        random_ood.append(dist)

    reset_physics()

    print()
    so = [d for d in structured_ood if d < float('inf')]
    ro = [d for d in random_ood if d < float('inf')]
    print(f"  Structured OOD: {np.mean(so):.2f} ± {np.std(so):.2f}")
    print(f"  Random OOD:     {np.mean(ro):.2f} ± {np.std(ro):.2f}")

    # Degradation analysis
    if s_valid and so:
        s_degrade = (np.mean(so) - np.mean(s_valid)) / np.mean(s_valid) * 100
        print(f"  Structured degradation: {s_degrade:+.1f}%")

    world.close()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/runpod-volume/checkpoints/lejepa_object.ckpt")
    parser.add_argument("--model_weights", default="/runpod-volume/data/temporal_best.pt")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run(**vars(args))

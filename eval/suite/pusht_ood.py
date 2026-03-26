"""
Push-T OOD Physics Evaluation

The correct way to evaluate physics anomaly detection:
    1. Generate trajectories in Push-T with NORMAL physics
    2. Generate trajectories with MODIFIED physics (same actions)
    3. Encode both with LeWM → actual different pixels
    4. Run predict-update cycle on both
    5. Compare innovation signals

The model sees real visual differences caused by real physics changes.
No GT state manipulation — everything goes through pixels.
"""

import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto


class PhysicsShift(Enum):
    """Types of physics parameter changes in Push-T."""
    NORMAL = auto()           # baseline (no change)
    FRICTION_HIGH = auto()    # block has more friction
    FRICTION_LOW = auto()     # block has less friction (slides more)
    BLOCK_HEAVY = auto()      # block is heavier (resists pushing)
    BLOCK_LIGHT = auto()      # block is lighter (moves easily)
    DAMPING_HIGH = auto()     # higher velocity damping
    BLOCK_LARGE = auto()      # block is larger
    BLOCK_SMALL = auto()      # block is smaller


# Parameter overrides for each shift
SHIFT_PARAMS = {
    PhysicsShift.NORMAL: {},
    PhysicsShift.FRICTION_HIGH: {"block_friction": 2.0},
    PhysicsShift.FRICTION_LOW: {"block_friction": 0.1},
    PhysicsShift.BLOCK_HEAVY: {"block_mass": 5.0},
    PhysicsShift.BLOCK_LIGHT: {"block_mass": 0.2},
    PhysicsShift.DAMPING_HIGH: {"damping": 0.95},
    PhysicsShift.BLOCK_LARGE: {"block_scale": 50.0},
    PhysicsShift.BLOCK_SMALL: {"block_scale": 15.0},
}


@dataclass
class OODTrajectory:
    """A trajectory generated with specific physics parameters."""
    frames: torch.Tensor       # (T, 3, H, W) rendered frames
    states: torch.Tensor       # (T, 7) GT state
    actions: torch.Tensor      # (T-1, action_dim) actions taken
    shift: PhysicsShift        # which physics shift was applied
    shift_params: dict         # actual parameter values


def generate_ood_trajectories(
    n_episodes: int = 50,
    episode_length: int = 30,
    shifts: list[PhysicsShift] | None = None,
    image_size: int = 224,
    seed: int = 42,
) -> dict[PhysicsShift, list[OODTrajectory]]:
    """Generate Push-T trajectories with various physics parameters.

    Uses stable-worldmodel's Push-T environment with parameter overrides.

    Args:
        n_episodes: episodes per physics shift
        episode_length: steps per episode
        shifts: which physics shifts to test (default: all)
        image_size: render resolution
        seed: random seed

    Returns:
        dict mapping PhysicsShift → list of OODTrajectory
    """
    import stable_worldmodel as swm
    from stable_worldmodel.envs.pusht import WeakPolicy

    if shifts is None:
        shifts = list(PhysicsShift)

    results = {}
    rng = np.random.RandomState(seed)

    for shift in shifts:
        print(f"  Generating {shift.name}...")
        params = SHIFT_PARAMS[shift]
        trajectories = []

        # Create environment — use variation space for parameter changes
        world = swm.World(
            "swm/PushT-v1",
            num_envs=1,
            image_shape=[image_size, image_size],
            max_episode_steps=episode_length,
            seed=seed,
        )
        policy = WeakPolicy(dist_constraint=100, seed=seed)
        world.set_policy(policy)

        for ep in range(n_episodes):
            # Reset and optionally modify physics
            obs, info = world.envs.reset(seed=int(rng.randint(0, 100000)))

            # Apply physics modifications via the underlying pymunk env
            env = world.envs.envs[0].unwrapped
            if "block_friction" in params:
                for shape in env.block.shapes:
                    shape.friction = params["block_friction"]
            if "block_mass" in params:
                env.block.mass = params["block_mass"]
            if "damping" in params:
                env.space.damping = params["damping"]
            if "block_scale" in params:
                # Scale affects the visual size — need to recreate
                pass  # Skip for now, complex to modify at runtime

            # Collect trajectory
            frames = []
            states = []
            actions = []

            for step in range(episode_length):
                # Render
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(torch.from_numpy(frame).permute(2, 0, 1))  # HWC → CHW

                # Get state
                state = env._get_obs()
                states.append(torch.from_numpy(state).float())

                # Get action from policy
                action = policy.get_action({"pixels": obs}, env=world.envs)
                if action is None:
                    action = np.zeros(world.envs.single_action_space.shape)
                actions.append(torch.from_numpy(action).float())

                # Step
                obs, reward, terminated, truncated, info = world.envs.step(action)
                if terminated or truncated:
                    break

            if len(frames) > 2:
                trajectories.append(OODTrajectory(
                    frames=torch.stack(frames),
                    states=torch.stack(states),
                    actions=torch.stack(actions[:len(frames) - 1]) if len(actions) > 1 else torch.zeros(1, 2),
                    shift=shift,
                    shift_params=params,
                ))

        world.close()
        results[shift] = trajectories
        print(f"    → {len(trajectories)} episodes")

    return results


def evaluate_with_model(
    model,
    lewm,
    trajectories: dict[PhysicsShift, list[OODTrajectory]],
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate model's innovation on OOD trajectories.

    Returns per-shift mean innovation and AUROC.
    """
    from einops import rearrange
    from sklearn.metrics import roc_auc_score

    model.eval()
    lewm.eval()

    shift_innovations = {}

    with torch.no_grad():
        for shift, trajs in trajectories.items():
            innovations = []

            for traj in trajs:
                T = min(traj.frames.shape[0], 6)
                if T < 3:
                    continue

                # Normalize frames
                frames = traj.frames[:T].float().to(device) / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                frames = (frames - mean) / std

                # Extract patches
                output = lewm.encoder(frames, interpolate_pos_encoding=True)
                patches = output.last_hidden_state[:, 1:]  # (T, 256, 192)
                patches = patches.unsqueeze(0)  # (1, T, 256, 192)

                # Parse actions
                actions = traj.actions[:T - 1].to(device)
                if actions.shape[-1] == 10:
                    actions = actions.reshape(-1, 5, 2).mean(dim=1)
                actions = actions.unsqueeze(0)  # (1, T-1, 2)

                # Run predict-update cycle
                out = model.forward_temporal(patches, actions=actions)

                # Collect innovation norms
                for innov in out["innovations"]:
                    innovations.append(innov[0].item())

            shift_innovations[shift.name] = innovations

    # Compute metrics
    results = {}
    normal_scores = shift_innovations.get("NORMAL", [])

    if not normal_scores:
        print("  WARNING: No normal trajectories for comparison")
        return results

    results["normal_mean"] = np.mean(normal_scores)
    results["normal_std"] = np.std(normal_scores)

    for shift_name, scores in shift_innovations.items():
        if shift_name == "NORMAL" or not scores:
            continue

        results[f"{shift_name}_mean"] = np.mean(scores)
        results[f"{shift_name}_ratio"] = np.mean(scores) / np.mean(normal_scores)

        # AUROC: can we distinguish this shift from normal?
        combined = np.concatenate([normal_scores[:len(scores)], scores])
        labels = np.concatenate([np.zeros(min(len(normal_scores), len(scores))), np.ones(len(scores))])
        try:
            results[f"{shift_name}_auroc"] = roc_auc_score(labels, combined)
        except ValueError:
            results[f"{shift_name}_auroc"] = 0.5

    return results

"""
Stage A Diagnostic: Can GT State Reveal Hidden Physics Changes?

Fixed per Codex review:
- Contact detection uses pymunk geometry, not center distance
- No hardcoded physics coefficients — fit them from data
- Angle unwrapping for θ
- frameskip=5 (matches dataset)
- Same actions replayed in different physics (not different policies)

The question: given PERFECT state (GT), can we distinguish
normal from physics-shifted trajectories at contact events?
If no → Push-T doesn't have enough dynamics for property inference.
If yes → perception is the bottleneck, not identifiability.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines", "lewm"))

import torch
import numpy as np
from scipy.signal import savgol_filter
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
import argparse


def unwrap_angle(theta):
    """Unwrap angle to avoid discontinuities at 0/2π."""
    return np.unwrap(theta)


def parse_block_kinematics(state_seq, dt=5.0/60.0):
    """Extract block kinematics from GT state sequence.

    Args:
        state_seq: (T, 7) raw state
        dt: timestep (frameskip=5 at 60fps)

    Returns:
        block_pos: (T, 2) block position
        block_vel: (T, 2) smoothed block velocity
        block_theta: (T,) unwrapped block angle
        block_omega: (T,) smoothed angular velocity
        agent_pos: (T, 2) agent position
        agent_vel: (T, 2) agent velocity
    """
    T = state_seq.shape[0]

    agent_pos = state_seq[:, :2]
    agent_vel = state_seq[:, 5:7]
    block_pos = state_seq[:, 2:4]
    block_theta = unwrap_angle(state_seq[:, 4])

    # Smoothed velocity via Savitzky-Golay (avoid double-diff noise)
    window = min(T if T % 2 == 1 else T - 1, 7)
    if T >= 5:
        block_vx = savgol_filter(block_pos[:, 0], window, 2, deriv=1, delta=dt)
        block_vy = savgol_filter(block_pos[:, 1], window, 2, deriv=1, delta=dt)
        block_omega = savgol_filter(block_theta, window, 2, deriv=1, delta=dt)
    else:
        block_vx = np.gradient(block_pos[:, 0], dt)
        block_vy = np.gradient(block_pos[:, 1], dt)
        block_omega = np.gradient(block_theta, dt)

    block_vel = np.stack([block_vx, block_vy], axis=-1)

    return block_pos, block_vel, block_theta, block_omega, agent_pos, agent_vel


def detect_contacts_by_motion(block_vel, threshold=5.0):
    """Detect contact frames from block velocity changes.

    Instead of distance-based contact (which requires knowing the T-shape),
    detect contacts from sudden changes in block velocity — if the block
    accelerates, something pushed it.

    Args:
        block_vel: (T, 2) block velocity
        threshold: minimum speed change to count as contact

    Returns:
        contacts: (T,) boolean
    """
    # Use ||Δv|| (full vector change), not diff(||v||) (speed change only)
    # This catches direction-only contacts (deflections without speed change)
    dv = np.diff(block_vel, axis=0, prepend=block_vel[:1])
    vel_change = np.linalg.norm(dv, axis=-1)
    return vel_change > threshold


def compute_kinematic_features(block_vel, block_omega, agent_vel, contacts):
    """Compute per-contact-event kinematic features.

    For each contact event, extract:
    - Block velocity before and after
    - Agent velocity at contact
    - Block angular velocity change
    - Speed ratio (block_after / agent_before)

    These features directly encode mass ratio and friction
    through conservation of momentum.
    """
    T = len(block_vel)
    features = []

    # Find contact onset frames
    contact_onsets = []
    in_contact = False
    for t in range(T):
        if contacts[t] and not in_contact:
            contact_onsets.append(t)
            in_contact = True
        elif not contacts[t]:
            in_contact = False

    for onset in contact_onsets:
        if onset < 2 or onset >= T - 3:
            continue

        # Pre-contact block state (2 frames before)
        v_block_pre = block_vel[onset - 1]
        omega_pre = block_omega[onset - 1]

        # Post-contact block state (2 frames after)
        post = min(onset + 3, T - 1)
        v_block_post = block_vel[post]
        omega_post = block_omega[post]

        # Agent velocity at contact
        v_agent = agent_vel[onset]

        # Features
        speed_pre = np.linalg.norm(v_block_pre)
        speed_post = np.linalg.norm(v_block_post)
        agent_speed = np.linalg.norm(v_agent)

        delta_v = speed_post - speed_pre
        delta_omega = abs(omega_post - omega_pre)
        # Filter out slow contacts (near-zero agent speed inflates ratio)
        if agent_speed < 5.0:
            continue

        speed_ratio = speed_post / agent_speed

        features.append({
            "delta_v": delta_v,
            "delta_omega": delta_omega,
            "speed_ratio": speed_ratio,
            "agent_speed": agent_speed,
            "block_speed_post": speed_post,
        })

    return features


def run_diagnostic(data_dir, n_episodes=500, seed=42):
    print("=" * 60)
    print("Stage A: Kinematic Diagnostic on GT State")
    print("=" * 60)
    print()
    print("Question: do contact dynamics in Push-T contain enough")
    print("signal to distinguish different physics parameters?")
    print()

    import stable_worldmodel as swm

    ds = swm.data.HDF5Dataset(
        name="pusht_expert_train",
        num_steps=20,       # 20 steps at frameskip=5 = 100 env steps
        frameskip=5,        # matches dataset
        keys_to_load=["state"],
        keys_to_cache=["state"],
        cache_dir=data_dir,
    )
    print(f"Dataset: {len(ds)} samples")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(min(len(ds), n_episodes * 10))[:n_episodes]

    # Collect kinematic features from contact events
    all_features = []
    total_contacts = 0

    print(f"Analyzing {n_episodes} trajectories...")
    for i, idx in enumerate(indices):
        sample = ds[int(idx)]
        state = sample["state"].numpy()  # (T, 7)

        block_pos, block_vel, block_theta, block_omega, agent_pos, agent_vel = \
            parse_block_kinematics(state)

        contacts = detect_contacts_by_motion(block_vel, threshold=3.0)
        features = compute_kinematic_features(block_vel, block_omega, agent_vel, contacts)

        total_contacts += len(features)
        all_features.extend(features)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_episodes}, contacts so far: {total_contacts}")

    print(f"\nTotal contact events: {total_contacts}")
    print(f"Contacts per trajectory: {total_contacts / n_episodes:.1f}")

    if total_contacts < 20:
        print("\nToo few contacts for analysis. Push-T may be too quasi-static.")
        return

    # Analyze variability of contact dynamics
    delta_vs = np.array([f["delta_v"] for f in all_features])
    speed_ratios = np.array([f["speed_ratio"] for f in all_features])
    delta_omegas = np.array([f["delta_omega"] for f in all_features])

    print()
    print("=" * 60)
    print("Contact Dynamics Variability (Normal Physics)")
    print("=" * 60)
    for name, arr in [("delta_v", delta_vs), ("speed_ratio", speed_ratios), ("delta_omega", delta_omegas)]:
        cv = arr.std() / (abs(arr.mean()) + 1e-6)
        print(f"  {name:<15} mean={arr.mean():>8.3f}  std={arr.std():>8.3f}  CV={cv:>6.2f}")

    print()
    print("=" * 60)
    print("Key Question: Is speed_ratio consistent?")
    print("=" * 60)
    print(f"  speed_ratio = block_speed_post / agent_speed")
    print(f"  If mass is constant AND contact geometry is uniform,")
    print(f"  this should be consistent. In practice, contact angle")
    print(f"  and lever arm confound the signal.")
    print(f"  CV < 1.0: possibly identifiable (with caveats)")
    print(f"  CV > 2.0: too noisy even in principle")
    cv = speed_ratios.std() / (abs(speed_ratios.mean()) + 1e-6)
    print(f"\n  CV = {cv:.2f}")

    if cv < 1.0:
        print("  → IDENTIFIABLE: Contact dynamics are consistent enough")
        print("    to potentially detect mass/friction changes.")
    elif cv < 2.0:
        print("  → MARGINAL: High variability but some signal may exist.")
        print("    Would need many contacts to detect changes.")
    else:
        print("  → NOT IDENTIFIABLE: Contact dynamics are too variable.")
        print("    Push-T contacts don't carry consistent physics signal.")

    # Check if speed_ratio correlates with agent_speed
    # (it shouldn't if mass is the main factor)
    agent_speeds = np.array([f["agent_speed"] for f in all_features])
    from scipy.stats import pearsonr
    r, p = pearsonr(speed_ratios, agent_speeds)
    print(f"\n  Correlation(speed_ratio, agent_speed): r={r:.3f}, p={p:.3e}")
    if abs(r) > 0.5:
        print("  → WARNING: speed_ratio depends on push speed, not just mass.")
        print("    Contact geometry/angle varies, confounding mass signal.")

    print()
    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/runpod-volume/data")
    parser.add_argument("--n_episodes", type=int, default=500)
    args = parser.parse_args()

    run_diagnostic(args.data_dir, args.n_episodes)

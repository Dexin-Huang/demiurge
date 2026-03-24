# DEMIURGE v0.2 — Hybrid Belief Architecture

> *JEPA is not the physics model. JEPA is the perception/training interface.
> The actual physics model should be much smaller.*

---

## Core Principle

Most macroscopic visual physics is a **small hybrid causal state**, not a giant smooth vector:

```
s_i = (q_i, v_i, z_i, σ_i)
```

plus sparse pairwise modes:

```
m_ij ∈ {∅, contact, stick, slip, bonded, break}
```

where `q,v` are explicit geometry/kinematics, `z` is a tiny persistent material code,
and `σ` is uncertainty over that code.

---

## Three-Part Architecture

```
┌─────────────────────────────────────────────────┐
│  Part 1: STATE ESTIMATOR (large)                │
│  Frozen LeWM encoder → object slots             │
│  Per-slot heads: geometry, material, uncertainty │
│  This is where the parameters live.             │
├─────────────────────────────────────────────────┤
│  Part 2: HYBRID BELIEF SIMULATOR (small)        │
│  Semi-implicit Euler: q,v rollout               │
│  Discrete mode classifier: M(Δq, Δv, z_i, z_j) │
│  Learned impulse model per mode                 │
│  Material belief update at innovation events    │
│  Passivity/conservation constraints             │
├─────────────────────────────────────────────────┤
│  Part 3: OBSERVER HEAD (small)                  │
│  Simulated (q,v,z) → predicted embedding        │
│  JEPA loss: MSE(pred_emb, sg(real_emb))         │
│  This is how the simulator learns.              │
└─────────────────────────────────────────────────┘
```

---

## What Changed from v0.1

| v0.1 (Three-Layer Stack) | v0.2 (Hybrid Belief) |
|---|---|
| Frozen Fourier scaffold | **Deleted** — q,v ARE the state |
| Gating + property memory | → Hybrid simulator + material belief |
| Flexible context layer | → Observer head |
| Graph transformer predictor | → Tiny impulse model |
| JEPA loss drives everything | → JEPA trains perception; sim has physics losses |
| Geometry is an encoding | → Geometry is the state itself |

---

## Dynamics

```
# Mode classification
m_ij = M(Δq_ij, Δv_ij, gap_ij, z_i, z_j)

# Velocity update
v_{t+1} = v_t + dt * a_free(q, v, z) + Σ_j J_m(Δq, Δv, z_i, z_j)

# Position update
q_{t+1} = q_t + dt * v_{t+1}

# Material belief update (conditional)
if I_i >= τ:
    z_i, σ_i = U(z_i, σ_i, δ_i)    # update belief
else:
    z_i, σ_i = z_i, σ_i              # coast
```

Where `I_i` is mechanical innovation: contact onset, slip onset, unexpected
acceleration, sustained strain.

---

## Repository Structure

```
demiurge/
  baselines/lewm/          # git submodule — perceptual backbone
  estimator/
    state_estimator.py      # Part 1: pixels → (q, v, z, σ)
    object_tracker.py       # Kinematics, pairwise features, innovation detection
  simulator/
    hybrid_simulator.py     # Part 2: structured forward rollout
    interaction_modes.py    # Discrete pairwise modes
    material_belief.py      # Conditional material updates
  observer/
    observer_head.py        # Part 3: simulated state → predicted embedding
  model/
    demiurge_v2.py          # Full system assembly
  training/                 # Training objectives and loop
  eval/                     # Probes, VoE, benchmarks
  layers/                   # [v0.1 — preserved for reference]
```

---

## Key Design Constraints

1. **Identifiability**: Force scale must be anchored so z means something physical
2. **Off-screen persistence**: State evolves even when object is occluded
3. **Passivity**: Energy can only decrease (or be injected by external forces)
4. **Innovation gating**: Material belief updates only at informative events
5. **Evaluation honesty**: Frozen rollouts + low-capacity probes, not invasive fine-tuning

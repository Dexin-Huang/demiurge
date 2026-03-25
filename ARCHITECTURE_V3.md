# DEMIURGE v0.3 — Physics-Constrained Slot Predictor

> *The representation is the hard part. The dynamics is just pairwise interactions.*
> *— Battaglia et al., 2016*

---

## Core Insight

JEPA is a strong perceptual encoder. Don't replace its predictor — augment it
with object structure and physics constraints in slot space.

The loss lives in **slot space**, not embedding space. Physics **constrains**
trajectories, it doesn't generate observations.

---

## Architecture

```
Frame t pixels (224×224)
    │
    ▼
Frozen LeWM ViT encoder
    │
    ├── Patch tokens (256 × 192)
    │        │
    │    Slot Attention (K slots × D_slot)
    │        │
    │        ├── Static head → appearance_i (constant, carried forward)
    │        │
    │        └── Dynamic head → (q_i, v_i) per object
    │                │
    │         Interaction Network
    │          (pairwise GNN, à la Battaglia 2016)
    │                │
    │                ├── φ_R(s_i, s_j) → effect_ij    (per edge)
    │                ├── Σ effects → aggregate_i        (per node)
    │                ├── φ_O(s_i, aggregate_i) → Δv_i   (per node)
    │                │
    │                ├── Conservation constraints:
    │                │     energy ≤ energy_prev
    │                │     Σ impulses ≈ 0 (Newton 3)
    │                │
    │                └── predicted (q'_i, v'_i)
    │                        │
    │                 Slot Assembly:
    │                 slot'_i = [static_i, encode(q'_i, v'_i)]
    │                        │
    │                        ▼
    │               predicted_slot'_{t+1}
    │
    └── CLS token (192) ────────────────────────────────┐
                                                         │
Frame t+1 pixels (224×224)                               │
    │                                                    │
    ▼                                                    │
Frozen LeWM ViT encoder                                 │
    │                                                    │
    ├── Patches → Slot Attention → actual_slot_{t+1}     │
    │                                                    │
    └── CLS → actual_cls_{t+1} ──────────────────────────┘
                                                         │
                                              (for baseline comparison only)
```

---

## Losses

```
L = L_slot                    # primary: predict future slots
  + λ_contrast · L_contrast   # contrastive: closer to true future than negatives
  + λ_energy · L_energy        # conservation: energy can't increase
  + λ_newton · L_newton3       # Newton's 3rd: impulses sum to zero
  + λ_state · L_state          # auxiliary: estimated (q,v) matches GT when available
```

### L_slot (primary)
MSE between predicted and actual future slot embeddings.
Each slot is [static, dynamic]. Static is carried forward (not predicted).
Only the dynamic component is predicted by the Interaction Network.

### L_contrast (prevents copy shortcut)
InfoNCE: predicted slot should be closer to true next slot than to
slots from other timesteps or other trajectories.

### L_energy (conservation)
Soft penalty: energy(t+1) ≤ energy(t). Computed from (q, v) in the
dynamic component. ReLU(E_{t+1} - E_t).

### L_newton3 (action-reaction)
For each interacting pair: ||effect_ij + effect_ji|| → 0.
Forces the interaction network to respect Newton's third law.

### L_state (supervised, optional)
When GT state is available (Push-T provides it):
MSE between estimated (q, v) and ground-truth positions/velocities.
This anchors the dynamic heads to produce physically meaningful state.

---

## Components

### 1. Slot Attention (perception → objects)

Input: LeWM ViT patch tokens (256 × 192)
Output: K slots × D_slot

Standard slot attention (Locatello 2020) over ViT patches:
- K = 3 for Push-T (agent + block + background)
- D_slot = 128
- 3 iterations
- Softmax over slots (competition for patches)

### 2. Static/Dynamic Heads (slot → factored state)

Per slot:
- **Static head**: MLP(D_slot → D_static). Encodes appearance (color, shape, texture).
  This is NOT predicted by physics — it's carried forward from the current frame.
- **Dynamic head**: MLP(D_slot → 4). Outputs (q_x, q_y, v_x, v_y).
  This IS predicted by the Interaction Network.

### 3. Interaction Network (dynamics)

The 2016 architecture, almost unchanged:

```python
# Per edge (pair of objects)
effect_ij = φ_R(cat[dynamic_i, dynamic_j, edge_attr_ij])

# Per node (aggregate incoming effects)
agg_i = Σ_j effect_ij

# Per node (update)
Δv_i = φ_O(cat[dynamic_i, agg_i])
v'_i = v_i + Δv_i
q'_i = q_i + dt * v'_i
```

φ_R and φ_O are 2-layer MLPs with ~64 hidden dim.
Edge attributes: relative position, relative velocity, distance.
Total params: ~10K. Tiny.

### 4. Dynamic Encoder (state → slot component)

MLP that maps predicted (q', v') back to the dynamic portion of the slot
embedding, so it can be compared with the actual future slot.

```python
dynamic_emb'_i = encode_dynamic(cat[q'_i, v'_i])  # (4,) → (D_dynamic,)
slot'_i = cat[static_i, dynamic_emb'_i]             # (D_static + D_dynamic,)
```

### 5. Material Codes (optional, for Phase 2)

Per-object z_i (8-dim) initialized from the slot, updated only at
mechanical innovation events (contact onset, unexpected acceleration).
Not needed for Phase 1 — add once basic dynamics work.

---

## What Stays from LeWM

- Frozen ViT-tiny encoder (5.5M params) — we use its patch tokens
- Projector + CLS pipeline — kept for baseline comparison
- LeWM's existing predictor — kept as the "unstructured" baseline (Condition A)

## What's New

- Slot Attention over ViT patches (~200K params)
- Static/Dynamic heads (~50K params)
- Interaction Network (~10K params)
- Dynamic encoder (~10K params)
- Conservation losses (0 params)
- **Total new: ~270K params**

---

## Ablation Structure

| Condition | Description |
|---|---|
| A: LeWM baseline | Unstructured CLS predictor (existing) |
| B: Slots only | Slot attention + slot prediction (no physics) |
| C: Slots + IN | Slot attention + Interaction Network dynamics |
| D: Slots + IN + conservation | Add energy + Newton 3 constraints |
| E: Full DEMIURGE v0.3 | Add material codes + innovation gating |

Each condition adds one component. The question at each step:
does structure help?

---

## Phase 1 Plan

1. **Slot attention on Push-T**: Extract K=3 slots from LeWM patches.
   Validate: do slots correspond to agent/block/background?
   (Check by comparing slot positions with GT state.)

2. **Static/dynamic disentanglement**: Train heads to split slots.
   Validate: does dynamic head recover (q, v)?
   (Supervised check against GT state.)

3. **Interaction Network**: Predict future dynamic state from current.
   Validate: does IN predict next (q, v) better than copying current?
   Loss in slot space, not CLS space.

4. **Conservation constraints**: Add energy + Newton 3.
   Validate: does energy decrease monotonically?
   Do long-horizon rollouts stay stable?

5. **Comparison**: Condition A (LeWM) vs Condition D (ours) on
   prediction accuracy, OOD transfer, VoE detection.

---

## Key Design Constraints

1. **Predict slots, not CLS.** A 128-dim per-object slot is tractable
   to predict from (q,v). A 192-dim global CLS is not.

2. **Static carried, dynamic predicted.** Appearance doesn't change.
   Physics only touches position and velocity.

3. **Interaction Network, not transformer.** Pairwise MLPs on edges.
   ~10K params. The dynamics is small because physics is small.

4. **Conservation as architecture, not hope.** Energy penalty and
   Newton 3 constraint are always-on losses, not learned.

5. **GT state supervision when available.** Push-T gives us (q, v).
   Use it to anchor the dynamic heads. This is not cheating —
   it's curriculum. Remove it later and see what survives.

---

## References

- Battaglia et al. 2016 — Interaction Networks
- Locatello et al. 2020 — Slot Attention
- Greydanus et al. 2019 — Hamiltonian Neural Networks
- Wu et al. 2023 — SlotFormer
- Nam et al. 2026 — C-JEPA
- Kipf et al. 2020 — C-SWM (contrastive structured world models)
- Baradel et al. 2020 — CoPhy (counterfactual physics)
- Battaglia et al. 2013 — Simulation as engine of physical understanding
- Maes et al. 2026 — LeWM (our backbone)

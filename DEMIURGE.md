# DEMIURGE
## Engineering Specification v0.2

> *The Demiurge does not create the forms. The forms are eternal and fixed. He learns how to apply them to matter.*

---

## 1. Core Hypothesis

Current JEPA-style world models entangle three distinct kinds of information in a single unstructured latent space:

- **Observable geometry** — position, velocity, spatial relations
- **Hidden intrinsic properties** — mass, friction, charge proxies
- **Flexible context** — appearance, texture, task-specific features

We hypothesize that explicitly separating these into three architectural layers with different update rules — frozen, slow, and fast — will produce measurably better sample efficiency and OOD transfer on physical reasoning tasks than parameter-matched baselines with unstructured latent spaces.

This is not a claim about AGI. It is a claim about physical domains, small data regimes, and the specific failure modes documented in IntPhys 2, CLEVRER, ComPhy, and Physion.

---

## 2. Backbone: LeWorldModel (LeWM)

**Do not build the JEPA backbone from scratch. Use LeWM.**

LeWM (Maes, Le Lidec, LeCun et al., 2026) solves the JEPA stability problem that previously required EMA targets, pretrained encoders, and complex multi-term losses to avoid representation collapse. It trains end-to-end from raw pixels with two loss terms: a next-embedding prediction loss and a Gaussian regularizer (SIGReg).

- Paper: https://arxiv.org/pdf/2603.19312v1
- Code: https://github.com/lucas-maes/le-wm
- ~15M parameters, trains on a single GPU in a few hours

**What LeWM gives us:**

A stable, lightweight JEPA backbone with an unstructured latent space that already encodes some physical structure — agent location, block location recover well from linear probes. Block angle is weaker. Hidden properties (mass, friction, charge) are not tested at all.

That gap — observable geometry partially encoded, hidden properties entirely absent — is exactly where DEMIURGE's three-layer stack is supposed to win.

**What DEMIURGE adds on top of LeWM:**

LeWM is Condition A in our ablation. DEMIURGE is Condition E. Everything between them is the experiment.

```
LeWM encoder (frozen after pretraining)
    ↓
Object tokenizer (dense tokens → K slots)
    ↓
Layer 1: Frozen geometric scaffold (DEMIURGE adds this)
Layer 2: Modulation — gating + property memory (DEMIURGE adds this)
Layer 3: Flexible context — LeWM's latent, now slot-structured
    ↓
Relational JEPA predictor (graph transformer over slots)
```

**LeWM's existing probe results set the bar:**

| Quantity | LeWM linear probe r↑ |
|---|---|
| Agent location | 0.974 |
| Block location | 0.986 |
| Block angle | 0.902 |
| Mass / friction / hidden properties | not tested |

DEMIURGE's scaffold should match or exceed these on observable quantities, and open the hidden-property column that LeWM never measures.

---

## 3. Architecture: The Three-Layer Stack

```
┌─────────────────────────────────────────────────────┐
│  LAYER 3 — NURTURE (fast)                           │
│  Flexible context cᵢₜ per slot                      │
│  Standard gradient descent                          │
│  Updated every step                                 │
├─────────────────────────────────────────────────────┤
│  LAYER 2 — MODULATION (slow)                        │
│  Gating weights α, β learned from scene context     │
│  Hidden property memory pᵢ per object               │
│  Regularized to change slowly — intrinsic property  │
│  Updated across tasks, not within a task            │
├─────────────────────────────────────────────────────┤
│  LAYER 1 — NATURE (frozen)                          │
│  Fixed geometric scaffold                           │
│  Unary + pairwise physical features                 │
│  Fixed Fourier/RBF encoding                         │
│  Zero trainable weights. Never updated.             │
└─────────────────────────────────────────────────────┘
```

**Key constraint:** Each layer communicates only with adjacent layers. Layer 3 cannot directly modify Layer 1. Experience shapes the phenotype, modulates expression, but cannot rewrite the genome.

---

## 4. Component Specification

### 4.1 Layer 1 — Nature (Frozen Geometric Scaffold)

No trainable weights in this layer. Ever.

**Unary features per object i at time t:**

| Feature | Description |
|---|---|
| `pos_i` | Normalized (x, y) position in frame |
| `ray_i` | Camera ray direction (monocular depth proxy) |
| `scale_i` | Bounding box scale |
| `vel_i` | Velocity (finite difference over frames) |
| `acc_i` | Acceleration |
| `t` | Timestep, normalized |

**Pairwise features per object pair (i, j) at time t:**

| Feature | Description |
|---|---|
| `Δpos_ij` | Relative displacement vector |
| `dist_ij` | Euclidean distance |
| `Δvel_ij` | Relative velocity |
| `contact_ij` | Binary overlap/contact cue |
| `depth_order_ij` | Which object is in front |
| `ttc_ij` | Time-to-contact estimate |

**Encoding:**

```python
def encode_geometric(features: np.ndarray) -> np.ndarray:
    # Fixed Fourier features — no learned parameters
    freqs = 2 ** np.arange(0, L) * np.pi  # L frequency bands
    encoded = np.concatenate([
        np.sin(freqs[None, :] * features[:, None]),
        np.cos(freqs[None, :] * features[:, None])
    ], axis=-1)
    return encoded  # shape: [num_features, 2L]
```

Use L=8 frequency bands as default. The encoding is deterministic and fixed at initialization.

**Why not global (x, y, z, t)?**

Global coordinates invite camera-frame shortcuts. The model learns "ball is at pixel 240,180" instead of "ball is 3 units left of the wall." Relative and relational features generalize across viewpoints and scenes. Physics is relational.

---

### 4.2 Layer 2 — Modulation (Slow Property Memory)

Two components: gating and hidden property memory.

**Gating:**

```
g̃ᵢₜ = αᵢₜ ⊙ φ(gᵢₜ)        # gated unary geometric features
r̃ᵢⱼₜ = βᵢⱼₜ ⊙ φ(rᵢⱼₜ)     # gated pairwise geometric features

αᵢₜ, βᵢⱼₜ ∈ [0,1]^d        # learned from scene context
```

Gates are produced by a small MLP conditioned on the current scene context. They allow the model to suppress geometric priors when the task doesn't need them and amplify them when it does.

**Hidden property memory pᵢ:**

A per-object vector that accumulates evidence about intrinsic properties (mass, friction, elasticity proxies) that are not directly observable in a single frame.

Update rule:
```
pᵢ(t) = (1 - γ) · pᵢ(t-1) + γ · f(interaction_history_i)
```

Where `γ` is small (e.g., 0.05) to enforce slow change. Regularization loss:

```
L_slow = ||pᵢ(t) - pᵢ(t-1)||²
```

This forces `pᵢ` to behave like an intrinsic property rather than a framewise feature. Mass doesn't change frame to frame. This layer learns to respect that.

---

### 4.3 Layer 3 — Nurture (Flexible Context)

Per-slot context vector `cᵢₜ` — appearance, texture, anything not captured by geometry or intrinsic properties. Updated with standard gradient descent. No special constraints.

**Slot state:**

```
sᵢₜ = [cᵢₜ, pᵢ, g̃ᵢₜ]
```

Context (fast) + property memory (slow) + gated geometry (frozen base, learned gate).

---

### 4.4 Object Tokenizer

Converts dense video tokens into K tracked slots + background.

**For initial experiments:** Use simulator masks or tracked proposals. Unsupervised slot discovery adds noise that obscures the inductive bias question. Solve the scaffold question first, then solve slot discovery.

**For later experiments:** Slot Attention over dense V-JEPA features.

K = 8 slots as default. Background = one additional slot.

---

### 4.5 Relational JEPA Predictor

Graph transformer over slots:

```
mᵢⱼₜ = ψ(cᵢₜ, cⱼₜ, pᵢ, pⱼ, r̃ᵢⱼₜ)          # message between objects
ŝₜ₊ₕ = P({c≤ₜ, p, g̃≤ₜ, m≤ₜ})               # predict future slot states
```

If reliable 3D geometry is available: replace graph transformer with SE(3)-equivariant attention block.

Add **object-level masking** (C-JEPA style): one object's future must be inferred from the others. This forces the model to use relational features, not just per-object extrapolation.

Multi-horizon prediction at h ∈ {1, 2, 4, 8} frames.

---

## 5. Training Objective

```
L = L_JEPA(ŝₜ₊ₕ, sg(sᵀₜ₊ₕ))     # main JEPA latent prediction loss
  + λ₁ · L_geom                   # auxiliary: geometric scaffold accuracy
  + λ₂ · L_slow_prop              # slow property regularization
  + λ₃ · L_gate_sparse            # gate sparsity — don't attend to everything
  + λ₄ · L_contact                # auxiliary: binary contact prediction
```

**Default weights:** λ₁=0.1, λ₂=0.05, λ₃=0.01, λ₄=0.1

**Contact loss is important:** Contact is where geometry and hidden properties meet. A ball bouncing depends on both where it is (geometry) and how elastic it is (hidden property). This loss forces both layers to coordinate.

---

## 6. Ablation Structure

Run all five in order. Each adds exactly one component. Budget must be parameter-matched across conditions. **Condition A is LeWM out of the box** — do not reimplement it, clone the repo and run it directly.

| Condition | Description | Backbone |
|---|---|---|
| A: LeWM baseline | Unstructured latent, no scaffold | LeWM as-is |
| B: LeWM + learnable geometry | Geometric features added, fully trainable | LeWM + trainable scaffold |
| C: LeWM + frozen geometry | Geometric features, frozen (Layer 1 only) | LeWM + frozen scaffold |
| D: LeWM + frozen geometry + gates | Add modulation gates (Layer 2 gating) | LeWM + Layer 1 + gates |
| E: Full DEMIURGE | Frozen scaffold + gating + property memory | Complete stack |

**The question at each step:** Does freezing outperform learning? Does gating outperform no gating? Does property memory close the hidden-property gap that LeWM never measures?

If B beats C — the frozen prior hypothesis is wrong. The model needs to adapt geometry, not preserve it. That's a real result.

**LeWM's known ceiling to beat:**

LeWM already struggles on visually complex 3D tasks (OGBench-Cube) where DINO-WM's large-scale pretraining gives it richer visual priors. DEMIURGE's bet is that explicit frozen structure closes that gap without needing orders-of-magnitude more pretraining data.

---

## 7. Evaluation Benchmarks

| Benchmark | What it tests |
|---|---|
| PHYRE | Sample-efficient physical intervention. Template/cross-template generalization. |
| CLEVRER | Causal, predictive, counterfactual reasoning about collisions |
| ComPhy | Hidden physical properties (mass, charge) not directly visible — LeWM not tested here |
| IntPhys 2 | Complex scenes, near-chance for most SOTA models |
| Push-T (LeWM's env) | Direct apples-to-apples comparison on LeWM's home turf |

**Primary metrics:**
- Sample efficiency curve (accuracy vs. number of training examples)
- OOD transfer: change gravity, friction, object mass — hold architecture fixed
- Probe accuracy: match or exceed LeWM's probe results on observable quantities, open the hidden-property column

**Probe targets (must beat LeWM's bar):**

| Quantity | LeWM r↑ (target to beat) | DEMIURGE target |
|---|---|---|
| Agent location | 0.974 | ≥ 0.974 |
| Block location | 0.986 | ≥ 0.986 |
| Block angle | 0.902 | ≥ 0.950 |
| Mass proxy | not tested | measurable |
| Friction proxy | not tested | measurable |

**Violation-of-expectation test (LeWM's paradigm, extended):**

LeWM tests surprise at color change and teleportation. Run the same test but add physically implausible dynamics: wrong mass ratio, wrong friction, wrong gravity. LeWM has no explicit physical prior — it should fail to flag these. DEMIURGE's frozen scaffold should catch them because the dynamics will contradict the geometric predictions.

This is the cleanest single test of whether the scaffold is load-bearing.

---

## 8. Build Order

**Phase 0 — Get LeWM running (2-3 days)**

```bash
git clone https://github.com/lucas-maes/le-wm
```

Run LeWM on Push-T. Reproduce their probe results. Confirm you can extract per-frame latents. This is your Condition A baseline. Do not proceed until this works.

**Phase 1 — Verify the scaffold (1 week)**

2D pymunk sim. Bouncing ball, gravity, one wall. Ground truth position and velocity available.
- Build Layer 1 encoding on top of LeWM latents
- Run probe test: does frozen scaffold encode (x, y, vel) correctly?
- If not, debug encoding before proceeding. Nothing downstream is valid until this passes.

**Phase 2 — Baseline comparisons (1-2 weeks)**

Run ablations A, B, C on PHYRE and Push-T.
- Does C (frozen) beat B (learnable)?
- This is the core question. Everything else depends on it.

**Phase 3 — Add modulation (1-2 weeks)**

Run ablations D, E.
- Do gates improve OOD transfer?
- Does property memory produce measurable signal on ComPhy hidden-property tasks?

**Phase 4 — VoE test**

Run violation-of-expectation on implausible physics (wrong mass, wrong gravity).
- Does DEMIURGE flag these as surprising where LeWM doesn't?
- This is the single most compelling qualitative result.

**Phase 5 — Replace simulator geometry with estimated geometry**

Swap ground-truth coordinates with estimated 2.5D tracks from pixels (depth estimator + optical flow).
- How much of the gain survives?
- This determines real-world applicability.

---

## 9. Repository Structure

```
demiurge/
  baselines/
    lewm/                  # git submodule: github.com/lucas-maes/le-wm
                           # Condition A — do not modify
  layers/
    nature.py              # frozen geometric scaffold — no optimizer touches this
    modulation.py          # gating + property memory
    nurture.py             # flexible slot context
  model/
    tokenizer.py           # LeWM dense tokens → object slots
    predictor.py           # graph transformer / SE3 attention
    demiurge.py            # full stack assembly, wraps LeWM encoder
  training/
    objectives.py          # all loss components including SIGReg from LeWM
    trainer.py             # training loop, layer-specific lr/freeze logic
  eval/
    probes.py              # linear probes — must match LeWM probe format
    voe.py                 # violation-of-expectation test, extended from LeWM
    benchmarks.py          # PHYRE, CLEVRER, ComPhy, IntPhys2, Push-T wrappers
  experiments/
    ablations/
      A_lewm_baseline/     # LeWM as-is, no modifications
      B_learnable_geom/
      C_frozen_geom/
      D_frozen_plus_gates/
      E_full_demiurge/
    prompt.md              # autoRubric entry point
```

**Critical:** LeWM lives as a submodule. Never copy-paste its code. If LeWM updates, pull the submodule. Your changes sit entirely in `layers/`, `model/`, and the wrapping logic in `demiurge.py`.

---

## 10. AutoRubric Integration

The autoRubric loop (builder + critic with evolving rubric) is the evaluation framework.

`prompt.md` for the loop:

```markdown
# Goal

A JEPA world model with three explicit layers:
1. Frozen geometric scaffold (nature)
2. Learned gating + slow property memory (modulation)  
3. Flexible slot context (nurture)

The model must outperform parameter-matched unstructured JEPA on:
- Sample efficiency on PHYRE (fewer examples to reach 70% accuracy)
- OOD transfer when physical parameters change (gravity, friction, mass)
- ComPhy hidden property inference

The frozen scaffold must encode ground-truth geometry — verified by linear probe.
The property memory must change slowly — verified by temporal autocorrelation.
The gates must not be trivially always-on or always-off — verified by entropy of α, β.
```

The critic will find gaps. The builder will close them. The rubric converges when neither can surprise the other.

---

## 11. Known Failure Modes to Watch For

**Layer 3 routes around Layer 1.** If the flexible context dims can absorb the prediction signal more easily than respecting the geometric scaffold, gradient descent will route around it. Monitor: are the geometric dims actually used by the predictor, or is attention weight on them near zero?

**Gates collapse.** α, β learn to be always 1 (ignore the prior) or always 0 (suppress everything). Monitor gate entropy. Add sparsity loss if needed.

**Property memory doesn't slow down.** If γ is too high, pᵢ behaves like a fast feature. Verify temporal autocorrelation of pᵢ is high (>0.9 across 10 frames).

**Simulator-to-pixel gap.** Ground-truth geometry is clean. Estimated 2.5D tracks are noisy. The gap between Phase 3 and Phase 4 is likely large. Don't overclaim from simulator results.

---

## 12. What Success Looks Like

Not "state of the art on benchmark X."

Success is: the ablation tells a coherent story. Frozen beats learnable. Gating improves transfer. Property memory produces signal on hidden-property tasks that LeWM cannot measure at all. The VoE test catches physically implausible dynamics that LeWM misses.

If any step in that story breaks — that's also success. You learned something true about whether physical priors help or get in the way.

The truth will be in the failure modes.

---

*Codename: DEMIURGE — the craftsman who imposes eternal geometric forms onto matter, without creating the forms themselves.*

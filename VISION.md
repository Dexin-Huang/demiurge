# DEMIURGE — North Star Vision

> *A world model is not a video predictor.*
> *It is a learned generative simulator of causal structure.*

---

## The Stack

```
PERCEIVE          →  STRUCTURE         →  SIMULATE         →  IMAGINE/ACT
(objects + state)    (rules + relations)   (run rules forward)  (counterfactuals + planning)
```

## The Developmental Sequence

Humans learn physics incrementally. Violations at each stage drive learning
the next stage. The system should follow the same curriculum:

| Stage | Capability | What's Learned | Learning Signal |
|-------|-----------|----------------|-----------------|
| 1 | Objects persist | Slot permanence | Tracking loss |
| 2 | Objects move smoothly | Continuity, dynamics | Prediction error |
| 3 | Surprise drives exploration | "Something unexpected — investigate" | Innovation signal |
| 4 | Interactions transfer properties | Momentum, contact rules | Active probing |
| 5 | Surfaces have properties | Friction, support | Repeated interaction |
| 6 | Materials have intrinsic properties | Mass, elasticity | Counterfactual comparison |
| 7 | Laws generalize across scenes | Transfer, compositionality | Multi-scene consistency |

**We are at Stage 2. The frontier is Stage 3.**

---

## The Key Insight

The OOD evaluation (March 2026) showed that passive observation cannot
detect hidden physics changes (AUROC ≈ 0.5 on mass/friction shifts).
This matches the cognitive science: infants don't learn mass from watching.
They learn from **active exploration after surprise.**

The predict-update cycle gives us the surprise signal (innovation).
The missing piece is: **using that surprise to drive targeted interaction
that resolves the ambiguity.**

---

## What We Proved

1. Slot attention extracts objects from frozen ViT patches (position r=0.96)
2. Interaction Network predicts dynamics (5.9K→272K params, beats copy baseline)
3. Predict-update cycle produces meaningful innovation signal
   (decreases 6.0→1.3 over 7 steps — Kalman convergence)
4. Innovation detects action perturbations (1.85x surprise, p=3e-241)
5. Innovation does NOT detect hidden physics changes from passive observation
   (AUROC ≈ 0.5 — negative result, but informative)

## What We Learned

- **v0.1** (three-layer stack on CLS): abandoned — wrong architecture
- **v0.2** (hybrid sim predicting embeddings): failed — can't predict 192-dim from 8 numbers
- **v0.3** (slot predictor with predict-update): working — the right framework
- **OOD passive detection**: failed — matches cognitive science prediction
- **The math for property inference is simple** (conservation of momentum, F=ma)
- **The bottleneck is perception** — and modern ViTs close that gap

## The Path Forward

Not "better anomaly detection." Not "bigger model."

**A system that uses surprise to discover physical laws, one stage at a time.**

Stage 3 is the next milestone: when the model is surprised (high innovation),
it should be able to **actively probe** the world to resolve the ambiguity.
"The block moved less than expected — is it heavier, or was the push weaker?
Let me push it again with a known force to find out."

This is the difference between a pattern matcher and a physicist.

---

## Architecture (current: v0.3, 622K params on frozen 18M LeWM)

```
Frozen LeWM ViT → patch tokens → SlotAttention (4 slots)
                                      ↓
                              SlotDecomposer (static + dynamic)
                                      ↓
                              InteractionNetwork (2 msg passes, action-conditioned)
                                      ↓
                              Predict-update cycle (Kalman-style)
                                      ↓
                              Innovation = corrected - predicted
```

## References

### Core
- Battaglia et al. 2013 — Simulation as engine of physical understanding
- Baillargeon 1987 — Object permanence, VoE paradigm
- Stahl & Feigenson 2015 — Surprise drives infant learning
- Spelke 1990 — Core knowledge principles

### Architecture
- Locatello et al. 2020 — Slot Attention
- Battaglia et al. 2016 — Interaction Networks
- Greydanus et al. 2019 — Hamiltonian Neural Networks
- Maes et al. 2026 — LeWM (our backbone)

### The Perception Bottleneck
- Wu et al. 2015 — Galileo (CNN + physics engine, failed on real data)
- Bhat et al. 2002 — Physical parameters from tracked trajectories
- Davis et al. 2015 — Material properties from vibration in video

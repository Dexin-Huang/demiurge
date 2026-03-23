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

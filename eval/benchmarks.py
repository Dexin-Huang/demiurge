"""
Benchmark Evaluation Wrappers

PHYRE, CLEVRER, ComPhy, IntPhys 2, Push-T
"""

from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Standard result format for all benchmarks."""
    benchmark: str
    condition: str
    metrics: dict[str, float]
    sample_sizes: list[int] | None = None  # for sample efficiency curves
    sample_metrics: dict[int, dict[str, float]] | None = None


class PHYREEvaluator:
    """PHYRE benchmark: sample-efficient physical intervention.

    Evaluates within-template and cross-template generalization.
    Reports AUCCESS metric at various sample sizes.
    """

    SAMPLE_SIZES = [10, 50, 100, 500, 1000, 5000]

    def __init__(self, split: str = "cross_template"):
        self.split = split
        # TODO: Initialize PHYRE environment
        # import phyre

    def evaluate(self, model, condition: str) -> BenchmarkResult:
        """Run PHYRE evaluation."""
        raise NotImplementedError("PHYRE evaluation — implement in Phase 2")


class PushTEvaluator:
    """Push-T evaluation using LeWM's native environment.

    Direct apples-to-apples comparison on LeWM's home turf.
    """

    def __init__(self):
        # TODO: Initialize Push-T via stable_worldmodel
        pass

    def evaluate(self, model, condition: str) -> BenchmarkResult:
        """Run Push-T evaluation."""
        raise NotImplementedError("Push-T evaluation — implement in Phase 2")


class ComPhyEvaluator:
    """ComPhy benchmark: hidden physical property inference.

    Tests mass, charge proxies that are not directly visible.
    This is where property memory (Layer 2) should shine.
    """

    def __init__(self):
        # TODO: Load ComPhy dataset
        pass

    def evaluate(self, model, condition: str) -> BenchmarkResult:
        """Run ComPhy evaluation."""
        raise NotImplementedError("ComPhy evaluation — implement in Phase 3")


class CLEVREREvaluator:
    """CLEVRER: causal, predictive, counterfactual reasoning about collisions."""

    def __init__(self):
        pass

    def evaluate(self, model, condition: str) -> BenchmarkResult:
        raise NotImplementedError("CLEVRER evaluation — implement in Phase 5")


class IntPhys2Evaluator:
    """IntPhys 2: complex scenes, near-chance for most SOTA models."""

    def __init__(self):
        pass

    def evaluate(self, model, condition: str) -> BenchmarkResult:
        raise NotImplementedError("IntPhys 2 evaluation — implement in Phase 5")

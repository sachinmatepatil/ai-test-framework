"""Shared test doubles. A seeded stochastic metric lets us prove the
non-determinism handling (reproducibility, sampling, stdev) without any
network or real model."""

import random

from aiqa.core.metric import Metric
from aiqa.core.result import SampleScore
from aiqa.core.thresholds import Threshold


class FixedScoreMetric(Metric):
    """Returns a constant score. Deterministic; for runner/threshold wiring."""

    is_deterministic = True

    def __init__(self, score: float, threshold: Threshold):
        super().__init__(threshold)
        self._score = score

    async def _measure_once(self, tc, *, seed, ctx) -> SampleScore:
        return SampleScore(score=self._score, reason="fixed")


class SeededNoisyMetric(Metric):
    """Draws from a normal dist using the PER-UNIT seed. Same global seed =>
    identical samples => reproducible, even under concurrency."""

    is_deterministic = False

    def __init__(self, mean: float, sigma: float, threshold: Threshold):
        super().__init__(threshold)
        self._mean, self._sigma = mean, sigma

    async def _measure_once(self, tc, *, seed, ctx) -> SampleScore:
        rng = random.Random(seed)
        raw = rng.gauss(self._mean, self._sigma)
        return SampleScore(score=max(0.0, min(1.0, raw)), reason=f"draw={raw:.3f}")


class ExplodingMetric(Metric):
    """Always raises; proves error isolation (-> ERRORED, suite survives)."""

    is_deterministic = True

    def __init__(self):
        super().__init__(Threshold(value=0.5))

    async def _measure_once(self, tc, *, seed, ctx) -> SampleScore:
        raise RuntimeError("boom")
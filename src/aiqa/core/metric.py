"""The Metric abstraction = an assertion about an AI output.

 Design: subclasses implement only `_measure_once` (produce one score for one case). The base class owns everything hard reusable:
 - multi-sampling for stochastic metrics (non-determinism handling)
 - aggregation(mean,media/pass-rate)
 - thresholding with the tolerance/noise band
 - error isolation (a raising metric becomes an ERRORED result, never crashes the suite)
 - latency capture


 This is why a new metric 10-20 lines: it inherits all of the above


Async first: `_measure_once` is async so the same code path serves cheap deterministic checks and slow networked LLM-judge calls.

Deepeval mapping: `Metric` <-> `BaseMetric`; `_measure_once` <-> `measure`;
our `Threshold` <-> BaseMetric's `threshold` + `success` flag
 """

from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod

from .config import RunContext
from .result import MetricResult, SampleScore, Status
from .thresholds import Aggregation, Threshold
from .testcase import TestCase

class Metric(ABC):
    #Deteministic metrics are run ONCE regardless of n_samples (sampling a
    #pure function N times is wasted cost). Stochastic metrics (LLM judges)
    # run n_samples times so we can assert on the distribution, not a fluke.
    is_deterministic: bool = True

    def __init__(self, threshold: Threshold, *, name: str | None = None) -> None:
        self.threshold = threshold
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def _measure_once(self, tc: TestCase, *, seed:int, ctx: RunContext) -> SampleScore:
        """Return one raw score in [0, 1] for this case. Subclasses implement.

        `seed` is the deterministic per-(case,metric,sample) seed: stochastic metrics MUST use it
        (not a global RNG) so runs are reproducible.
        """

    async def evaluate(self, tc: TestCase, ctx: RunContext) -> MetricResult:
        """Full evaluation: sample -> Aggregate -> threshold -> result.
        Never raises; faults become ERRORED results.
        """
        n = 1 if self.is_deterministic else max(1, ctx.n_samples)
        start = time.perf_counter()
        try:
            samples = await self._collect_samples(tc, ctx, n)
        except Exception as exc:
            return MetricResult(
                metric_name=self.name,
                test_case_id=tc.id,
                status=Status.ERRORED,
                threshold=self.threshold.describe(),
                error=f"{type(exc).__name__}: {exc}",
                n_samples=n,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        scores = [s.score for s in samples]
        agg = self._aggregate(scores)
        passed = self._verdict(scores, agg)
        spread = statistics.stdev(scores) if len(scores) > 1 else None

        return MetricResult(
            metric_name=self.name,
            test_case_id=tc.id,
            status=Status.PASSED if passed else Status.FAILED,
            threshold=self.threshold.describe(),
            score=agg,
            reason=samples[-1].reason,
            samples=scores,
            n_samples=n,
            start=spread,
            latency_ms=latency_ms,
            stdev=spread,
        )

    async def _collect_samples(
            self, tc: TestCase, ctx: RunContext, n: int
    ) -> list[SampleScore]:
        out: list[SampleScore] = []
        for i in range(n):
            seed = ctx.seed_for(tc.id, self.name, str(i))
            out.append(await self._measure_once(tc, seed=seed, ctx=ctx))
        return out

    def _aggregate(self, scores: list[float]) -> float:
        match self.threshold.aggregation:
            case Aggregation.MEAN:
                return statistics.fmean(scores)
            case Aggregation.MEDIAN:
                return statistics.median(scores)
            case Aggregation.PASS_RATE:
                hits = sum(1 for s in scores if self.threshold.passes(s))
                return hits / len(scores)

    def _verdict(self, scores: list[float], agg: float) -> bool:
        # PASS_RATE compares the fraction-passing against min_pass_rate;
        # MEAN/MEDIAN apply the threshold to the aggregated score directly.
        if self.threshold.aggregation is Aggregation.PASS_RATE:
            return agg >= self.threshold.min_pass_rate
        return self.threshold.passes(agg)
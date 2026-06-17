"""Tests for the Metric base: sampling, aggregation, verdict, error isolation,
and the determinism guarantee."""

import pytest

from aiqa.core.config import RunContext
from aiqa.core.result import Status
from aiqa.core.testcase import TestCase
from aiqa.core.thresholds import Aggregation, Direction, Threshold

from .doubles import ExplodingMetric, FixedScoreMetric, SeededNoisyMetric

CASE = TestCase(id="t1", input="q", actual_output="a", expected_output="a")


@pytest.mark.asyncio
class TestEvaluate:
    async def test_deterministic_metric_runs_once_even_with_many_samples(self):
        m = FixedScoreMetric(0.9, Threshold(value=0.7))
        res = await m.evaluate(CASE, RunContext(n_samples=10))
        assert res.n_samples == 1  # deterministic => sampling skipped
        assert res.status is Status.PASSED
        assert res.score == 0.9

    async def test_failing_score_yields_failed_not_errored(self):
        m = FixedScoreMetric(0.5, Threshold(value=0.7))
        res = await m.evaluate(CASE, RunContext())
        assert res.status is Status.FAILED  # a real defect, not a harness fault

    async def test_exception_is_isolated_as_errored(self):
        res = await ExplodingMetric().evaluate(CASE, RunContext())
        assert res.status is Status.ERRORED
        assert "boom" in res.error

    async def test_stochastic_metric_collects_n_samples_and_stdev(self):
        m = SeededNoisyMetric(0.8, 0.1, Threshold(value=0.6))
        res = await m.evaluate(CASE, RunContext(n_samples=20))
        assert res.n_samples == 20
        assert len(res.samples) == 20
        assert res.stdev is not None and res.stdev > 0

    async def test_same_seed_is_reproducible(self):
        m = SeededNoisyMetric(0.8, 0.1, Threshold(value=0.6))
        r1 = await m.evaluate(CASE, RunContext(seed=123, n_samples=15))
        r2 = await m.evaluate(CASE, RunContext(seed=123, n_samples=15))
        assert r1.samples == r2.samples  # identical draws => no flakiness

    async def test_different_seed_changes_draws(self):
        m = SeededNoisyMetric(0.8, 0.1, Threshold(value=0.6))
        r1 = await m.evaluate(CASE, RunContext(seed=1, n_samples=15))
        r2 = await m.evaluate(CASE, RunContext(seed=2, n_samples=15))
        assert r1.samples != r2.samples


@pytest.mark.asyncio
class TestAggregation:
    async def test_pass_rate_requires_min_fraction(self):
        # mean ~0.55, sits near the line; PASS_RATE asks "how often >= 0.55?"
        thr = Threshold(
            value=0.55, aggregation=Aggregation.PASS_RATE, min_pass_rate=0.9
        )
        m = SeededNoisyMetric(0.55, 0.3, thr)
        res = await m.evaluate(CASE, RunContext(n_samples=50))
        # ~half the draws clear 0.55, so a 90% pass-rate requirement fails.
        assert res.status is Status.FAILED

    async def test_median_robust_to_outliers(self):
        thr = Threshold(value=0.6, aggregation=Aggregation.MEDIAN)
        m = SeededNoisyMetric(0.8, 0.05, thr)
        res = await m.evaluate(CASE, RunContext(n_samples=11))
        assert res.status is Status.PASSED
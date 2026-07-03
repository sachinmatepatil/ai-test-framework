"""Tests for the Runner: result grouping, input-order preservation, aggregates,
and that concurrency does not break determinism."""

import pytest

from aiqa.core.config import RunContext
from aiqa.core.result import Status
from aiqa.core.runner import Runner
from aiqa.core.testcase import TestCase
from aiqa.core.thresholds import Threshold

from .doubles import ExplodingMetric, FixedScoreMetric, SeededNoisyMetric

CASES = [TestCase(id=f"c{i}", input=f"q{i}") for i in range(5)]


@pytest.mark.asyncio
class TestRunner:
    async def test_groups_results_by_case_in_input_order(self):
        m = FixedScoreMetric(0.9, Threshold(value=0.7))
        suite = await Runner().run(CASES, [m])
        assert [t.test_case_id for t in suite.test_results] == [c.id for c in CASES]
        assert suite.total == 5

    async def test_multiple_metrics_per_case(self):
        metrics = [
            FixedScoreMetric(0.9, Threshold(value=0.7)),
            FixedScoreMetric(0.4, Threshold(value=0.7)),
        ]
        suite = await Runner().run(CASES, metrics)
        first = suite.test_results[0]
        assert len(first.metric_results) == 2
        assert first.status is Status.FAILED  # one metric failed => case fails

    async def test_errored_metric_does_not_sink_suite(self):
        suite = await Runner().run(CASES, [ExplodingMetric()])
        assert suite.total == 5
        assert suite.errored_count == 5
        assert suite.failed_count == 0  # errors are not counted as failures

    async def test_aggregates_pass_rate(self):
        suite = await Runner().run(CASES, [FixedScoreMetric(0.9, Threshold(value=0.7))])
        assert suite.pass_rate == 1.0
        assert suite.mean_score("FixedScoreMetric") == pytest.approx(0.9)

    async def test_high_concurrency_is_still_deterministic(self):
        m = SeededNoisyMetric(0.7, 0.2, Threshold(value=0.5))
        ctx_a = RunContext(seed=99, n_samples=8, concurrency=16)
        ctx_b = RunContext(seed=99, n_samples=8, concurrency=1)
        # Same seed, wildly different parallelism => identical sample sets.
        suite_hi = await Runner(ctx_a).run(CASES, [m])
        suite_lo = await Runner(ctx_b).run(CASES, [m])
        hi = [t.metric_results[0].samples for t in suite_hi.test_results]
        lo = [t.metric_results[0].samples for t in suite_lo.test_results]
        assert hi == lo
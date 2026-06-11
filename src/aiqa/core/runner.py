"""The suite runner: evaluates (case x metric) units concurrently, with a
bounded concurrency limit, deterministic output ordering, and a single
SuiteResult out the other end.

Why async + semaphore: real metrics are I/O-bound (LLM API calls). We want
many in flight, but not unbounded (rate limits, cost). The semaphore caps
parallelism; ordering of *output* is restored deterministically afterward so
reports diff cleanly run-to-run.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from .config import RunContext
from .metric import Metric
from .result import MetricResult, SuiteResult, TestResult
from .testcase import TestCase


class Runner:
    def __init__(self, ctx: RunContext | None = None) -> None:
        self.ctx = ctx or RunContext()

    async def run(
        self, cases: list[TestCase], metrics: list[Metric]
    ) -> SuiteResult:
        sem = asyncio.Semaphore(self.ctx.concurrency)

        async def one(tc: TestCase, metric: Metric) -> tuple[str, MetricResult]:
            async with sem:
                return tc.id, await metric.evaluate(tc, self.ctx)

        tasks = [one(tc, m) for tc in cases for m in metrics]
        # evaluate() never raises, so gather needs no return_exceptions guard.
        pairs = await asyncio.gather(*tasks)

        # Regroup metric results under their case, preserving input order.
        by_case: dict[str, list[MetricResult]] = {tc.id: [] for tc in cases}
        for case_id, result in pairs:
            by_case[case_id].append(result)

        test_results = [
            TestResult(test_case_id=tc.id, metric_results=by_case[tc.id])
            for tc in cases
        ]

        return SuiteResult(
            run_id=self.ctx.run_id,
            seed=self.ctx.seed,
            test_results=test_results,
            started_at=self.ctx.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            metadata=self.ctx.metadata,
        )
"""Result objects. Immutable records of what happened, designed to be the single source of truth for reporting
,regression gating and defect creation.

Levels mirror the run hierarchy:
    SampleScore -> one raw measurement (before aggregation/thresholding)
    MetricResult -> one metric on one test case (after aggregation + verdict)
    TestResult -> all metrics on one test case
    SuiteResult -> the whole run, with aggregates + reproducibility metadata
"""

from __future__ import annotations

from ast import Suite
from  enum import Enum
import statistics
from symtable import Class

from pydantic import BaseModel, Field

class Status(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"
    #Errored is distinct from FAILED on purpose: a FAIL is a real defect in the system under test;
    #an ERROR is a fault in the harness/infra(timeout, bad parse)
    #Conflating them is a classic way to hide regressions or chase ghosts.

class SampleScore(BaseModel):
    """One raw measurement from a metric's `measure_once`"""

    score: float
    reason: str | None = None

class MetricResult(BaseModel):
    metric_name: str
    test_case_id: str
    status: Status
    score: float | None = None #aggregated score (None if erroed)
    threshold: str = "" #Human-readable threshold, e.g. "score>=0.7"
    reason: str | None = None
    error: str | None = None

    #Reproducibility + non-determinism evidence.
    samples: list[float] = Field(default_factory=list)
    n_samples: int = 1
    stdev: float | None = None
    seed: int | None = None
    latency_ms: float | None = None

    @property
    def passed(self) -> bool:
        return self.status is Status.PASSED

class TestResult(BaseModel):
    __test__ = False #Domain model, not a pytest test class

    test_case_id: str
    metric_results : list[MetricResult]

    @property
    def status(self) -> Status:
        if any(m.status is Status.ERRORED for m in self.metric_results):
            return Status.ERRORED
        if any(m.status is Status.FAILED for m in self.metric_results):
            return Status.FAILED
        return Status.PASSED

    @property
    def passed(self) -> bool:
        return self.status is Status.PASSED

class SuiteResult(BaseModel):
    run_id : str
    seed : int
    test_results : list[TestResult]
    started_at : str
    finished_at : str
    metadata : dict[str, object] = Field(default_factory=dict)

    # --- aggregates used by reporting and CI gating ---
    @property
    def total(self) -> int:
        return len(self.test_results)

    @property
    def passed_count(self) -> int:
        return sum(1 for t in self.test_results if t.status is Status.PASSED)

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.test_results if t.status is Status.FAILED)

    @property
    def errored_count(self) -> int:
        return sum(1 for t in self.test_results if t.status is Status.ERRORED)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total if self.total else 0.0

    def mean_score(self, metric_name: str) -> float | None:
        """Mean aggregated score for one metric across all cases (for baselines)."""
        scores = [
            m.score
            for t in self.test_results
            for m in t.metric_results
            if m.metric_name == metric_name and m.score is not None
        ]
        return statistics.fmean(scores) if scores else None

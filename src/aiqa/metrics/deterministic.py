""" A first concrete metric, to prove the engine end-to-end and demonstrate the
"<30 lines to add a metric" claim. The full deteministic suit (fuzzy, regex, JSON-schema, length/format contracts)
"""

from __future__ import annotations

from ..core.config import RunContext
from ..core.metric import Metric
from ..core.result import SampleScore
from ..core.testcase import TestCase
from ..core.thresholds import Direction, Threshold

class ExactMatchMetric(Metric):
    """1.0 if actual_output equals expected_output, else 0.0."""

    is_deterministic = True

    def __init__(self, *, case_sensitive: bool = True) -> None:
        super().__init__(Threshold(value=1.0, direction=Direction.HIGHER_IS_BETTER))
        self.case_sensitive = case_sensitive

    async def _measure_once(
        self, tc: TestCase, *, seed: int, ctx: RunContext
    ) -> SampleScore:
        actual = tc.actual_output or ""
        expected = tc.expected_output or ""
        if not self.case_sensitive:
            actual, expected = actual.lower(), expected.lower()
        hit = actual == expected
        return SampleScore(
            score=1.0 if hit else 0.0,
            reason="exact match" if hit else f"{actual!r} != {expected!r}",
        )
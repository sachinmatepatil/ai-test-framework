"""End-to-end smoke test of the example metric through the runner."""

import pytest

from aiqa.core.runner import Runner
from aiqa.core.result import Status
from aiqa.core.testcase import TestCase
from aiqa.metrics.deterministic import ExactMatchMetric
from aiqa.core.thresholds import Threshold, Direction


@pytest.mark.asyncio
async def test_exact_match_pass_and_fail():
    cases = [
        TestCase(id="hit", input="2+2", actual_output="4", expected_output="4"),
        TestCase(id="miss", input="2+2", actual_output="5", expected_output="4"),
    ]
    suite = await Runner().run(cases, [ExactMatchMetric()])
    by_id = {t.test_case_id: t for t in suite.test_results}
    assert by_id["hit"].status is Status.PASSED
    assert by_id["miss"].status is Status.FAILED
    assert "!=" in by_id["miss"].metric_results[0].reason


@pytest.mark.asyncio
async def test_case_insensitive_option():
    case = TestCase(id="ci", input="x", actual_output="Yes", expected_output="yes")
    suite = await Runner().run([case], [ExactMatchMetric(case_sensitive=False)])
    assert suite.test_results[0].status is Status.PASSED
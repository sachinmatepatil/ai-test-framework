# aiqa — AI/LLM Testing & Evaluation Framework

A production-grade, **testing-first** evaluation framework for AI systems —
LLMs, RAG pipelines, agents, MCP tools, and benchmarks. Built from scratch to
master the internals, then mapped onto industry-standard tooling (DeepEval,
RAGAS, LangSmith).

## Why this exists

Traditional tests assume determinism: same input → same output → `assert ==`.
LLMs break that contract — outputs are *samples from a distribution*. This
framework treats a result as something you assert about **statistically and
within tolerance**, with reproducible seeding so a red test means a real
regression, not noise.

## Features (Phase 1 — core engine)

- **`Metric`** — an assertion abstraction; a new metric is one method.
- **`TestCase`** — a typed, immutable test-data schema (mirrors DeepEval's `LLMTestCase`).
- **`Threshold`** — pass/fail policy with direction (higher/lower-is-better),
  a tolerance/noise band, and aggregation (mean / median / pass-rate).
- **Three-state results** — `PASSED` / `FAILED` / `ERRORED` (a real defect is
  not the same as a harness fault).
- **Deterministic seeding** — reproducible results even under async concurrency.
- **Async runner** — bounded concurrency for rate-limit and cost safety.
- **99% self-test coverage** — the framework tests itself.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Quickstart

```python
import asyncio
from aiqa.core.runner import Runner
from aiqa.core.testcase import TestCase
from aiqa.metrics.deterministic import ExactMatchMetric

cases = [
    TestCase(id="q1", input="2+2", actual_output="4", expected_output="4"),
    TestCase(id="q2", input="2+2", actual_output="5", expected_output="4"),
]
suite = asyncio.run(Runner().run(cases, [ExactMatchMetric()]))
print(f"pass rate: {suite.pass_rate:.0%}")   # pass rate: 50%
```

## Run the tests

```bash
python -m pytest -q --cov=src/aiqa --cov-report=term-missing
```

## Project structure
```
src/aiqa/
core/         # engine: testcase, thresholds, result, config, metric, runner
metrics/      # concrete metrics (Phase 1: exact match)
tests/unit/     # the framework's own tests
'''

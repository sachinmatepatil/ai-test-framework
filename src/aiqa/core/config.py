"""Run configuration and the determinism backbone.

The single most important function here is `seed_for`. Under asyncio
concurrency, tasks interleave non-deterministically, so a *shared* RNG produces
a *different draw order every run* -> flaky tests. We instead derive a stable,
independent seed for each (case, metric, sample) unit from the global seed.
Same global seed => identical results, regardless of scheduling.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_seed(base_seed: int, *parts: str) -> int:
    """Stable per-unit seed. Uses sha256 (NOT Python's salted hash()) so the
    value is identical across processes and runs."""
    key = f"{base_seed}:" + ":".join(parts)
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class RunContext:
    """Carries everything needed to reproduce a run."""

    seed: int = 42
    n_samples: int = 1  # samples per stochastic metric; 1 for deterministic
    concurrency: int = 8  # max metrics evaluated in parallel
    run_id: str = field(default_factory=lambda: derive_seed(0, _now_iso()).__str__())
    started_at: str = field(default_factory=_now_iso)
    metadata: dict[str, object] = field(default_factory=dict)

    def seed_for(self, *parts: str) -> int:
        return derive_seed(self.seed, *parts)
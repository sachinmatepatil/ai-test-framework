"""Pass/Fail decisioning: thresholds, directions and the noise band.

This module is where the central LLM-testing problem lives: a score is a *sample from a distribution*
so the pass/fail line must account for which direction is "good" and for measurement noise at the boundary.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

class Direction(str, Enum):
    """ Which way is 'better'

    HIGHER_IS_BETTER = relevancy, faithfulness, correctness (score >= threshold).
    LOWER_IS_BETTER = bias, hallucination-rate, toxicity (score <= threshold).
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"

class Aggregation(str, Enum):
    """ How to collapse N samples of stochastic metric into one verdict

    MEAN: assert on the average score (smooths noise; default).
    MEDIAN: assert on the median (robust to a single outlier sample)
    PASS_RATE: each sample is judged individually; pass if>= 'min_pass_rate' of sample
            pass. Use for "Must succeed k out of N times"
    """

    MEAN = "mean"
    MEDIAN = "median"
    PASS_RATE = "pass_rate"

class Threshold(BaseModel):
    """The pass/fail line for a single metric.

    `tolerance is the NOISE BAND, not a fudge factor. It exists to stop
    borderline scores from flapping pass/fail run-to-run (the #1 source of CI flakiness for LLM tests). It should be set from *measured* metric
    variance (e.g. ~2x the std-dev of repeated measurements), never guessed.
    A score withing `tolerance` of the line is treated as passing.
    """

    value: float
    direction: Direction = Direction.HIGHER_IS_BETTER
    tolerance: float = Field(default=0.0,ge=0.0)
    aggregation: Aggregation = Aggregation.MEAN
    min_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}

    def passes(self, score:float) -> bool:
        """Does a single (already-aggregated) score pass, within tolerance ? """
        if self.direction is Direction.HIGHER_IS_BETTER:
            return score >= self.value - self.tolerance
        return score <= self.value + self.tolerance

    def describe(self) -> str:
        op = ">=" if self.direction is Direction.HIGHER_IS_BETTER else "<="
        band = f" (±{self.tolerance})" if self.tolerance else ""
        return f"score {op} {self.value} {band} "

"""Tests for the pass/fail core: direction, tolerance/noise band, aggregation."""

from aiqa.core.thresholds import Aggregation, Direction, Threshold


class TestDirection:
    def test_higher_is_better_passes_above_line(self):
        t = Threshold(value=0.7, direction=Direction.HIGHER_IS_BETTER)
        assert t.passes(0.71)
        assert t.passes(0.70)
        assert not t.passes(0.69)

    def test_lower_is_better_passes_below_line(self):
        # e.g. toxicity: low is good
        t = Threshold(value=0.1, direction=Direction.LOWER_IS_BETTER)
        assert t.passes(0.05)
        assert t.passes(0.10)
        assert not t.passes(0.11)


class TestToleranceNoiseBand:
    def test_tolerance_absorbs_boundary_noise_higher(self):
        # 0.69 would fail a hard 0.70 line, but a 0.02 noise band rescues it.
        strict = Threshold(value=0.70)
        lenient = Threshold(value=0.70, tolerance=0.02)
        assert not strict.passes(0.69)
        assert lenient.passes(0.69)

    def test_tolerance_widens_lower_is_better(self):
        t = Threshold(value=0.10, direction=Direction.LOWER_IS_BETTER, tolerance=0.02)
        assert t.passes(0.12)  # within band
        assert not t.passes(0.13)


class TestDescribe:
    def test_describe_renders_band(self):
        assert "±0.02" in Threshold(value=0.7, tolerance=0.02).describe()
        assert ">=" in Threshold(value=0.7).describe()
        assert "<=" in Threshold(
            value=0.7, direction=Direction.LOWER_IS_BETTER
        ).describe()
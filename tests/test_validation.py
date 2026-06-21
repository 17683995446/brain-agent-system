"""Tests for input validation (Stage 6 quality)."""

import pytest
from brain_agent import Simulation, Signal
from brain_agent.scheduler import SchedulerConfig


def _get_json_path():
    import os
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "brain_tree_v5.json"),
        "/workspace/BrainAgentSystem/data/brain_tree_v5.json",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    pytest.skip("brain_tree_v5.json not found")


class TestSchedulerConfigValidation:
    """Test SchedulerConfig __post_init__ validation."""

    def test_valid_config(self):
        config = SchedulerConfig()
        assert config.dt_fast == 0.005

    def test_negative_dt_fast_raises(self):
        with pytest.raises(ValueError, match="dt_fast must be positive"):
            SchedulerConfig(dt_fast=-1.0)

    def test_zero_dt_fast_raises(self):
        with pytest.raises(ValueError, match="dt_fast must be positive"):
            SchedulerConfig(dt_fast=0.0)

    def test_negative_dt_medium_raises(self):
        with pytest.raises(ValueError, match="dt_medium must be positive"):
            SchedulerConfig(dt_medium=-0.01)

    def test_negative_dt_slow_raises(self):
        with pytest.raises(ValueError, match="dt_slow must be positive"):
            SchedulerConfig(dt_slow=-0.1)

    def test_dt_medium_less_than_dt_fast_raises(self):
        with pytest.raises(ValueError, match="dt_medium.*must be >= dt_fast"):
            SchedulerConfig(dt_fast=0.02, dt_medium=0.005)

    def test_dt_slow_less_than_dt_medium_raises(self):
        with pytest.raises(ValueError, match="dt_slow.*must be >= dt_medium"):
            SchedulerConfig(dt_medium=0.1, dt_slow=0.02)

    def test_negative_max_latency_raises(self):
        with pytest.raises(ValueError, match="max_latency_ms must be positive"):
            SchedulerConfig(max_latency_ms=-1.0)

    def test_negative_max_steps_raises(self):
        with pytest.raises(ValueError, match="max_steps must be >= 0"):
            SchedulerConfig(max_steps=-1)

    def test_zero_max_steps_allowed(self):
        config = SchedulerConfig(max_steps=0)
        assert config.max_steps == 0


class TestStepsValidation:
    """Test steps parameter validation in run() and iter_steps()."""

    def test_run_negative_steps_raises(self):
        sim = Simulation.from_json(_get_json_path())
        with pytest.raises(ValueError, match="steps must be >= 0"):
            sim.run(steps=-1)

    def test_iter_steps_negative_steps_raises(self):
        sim = Simulation.from_json(_get_json_path())
        with pytest.raises(ValueError, match="steps must be >= 0"):
            list(sim.iter_steps(steps=-1))

    def test_run_zero_steps_returns_empty(self):
        sim = Simulation.from_json(_get_json_path())
        results = sim.run(steps=0)
        assert len(results) == 0

    def test_iter_steps_zero_steps_returns_empty(self):
        sim = Simulation.from_json(_get_json_path())
        results = list(sim.iter_steps(steps=0))
        assert len(results) == 0

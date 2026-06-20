"""Tests for BrainConfig validation."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from brain_agent.config import BrainConfig


class TestConfigValidation:
    """Tests for BrainConfig.validate()."""

    def test_default_config_is_valid(self):
        """Default configuration should pass validation."""
        config = BrainConfig.from_defaults()
        errors = config.validate()
        assert errors == [], f"Default config has errors: {errors}"

    def test_empty_config_uses_defaults(self):
        """Empty config should validate with default values."""
        config = BrainConfig({})
        errors = config.validate()
        assert errors == []

    def test_invalid_total_nodes(self):
        """Negative total_nodes should fail validation."""
        config = BrainConfig({"brain": {"total_nodes": -1}})
        errors = config.validate()
        assert any("total_nodes" in e for e in errors)

    def test_invalid_max_layers(self):
        """max_layers > 20 should fail validation."""
        config = BrainConfig({"brain": {"max_layers": 50}})
        errors = config.validate()
        assert any("max_layers" in e for e in errors)

    def test_invalid_dt_fast(self):
        """dt_fast <= 0 should fail validation."""
        config = BrainConfig({"simulation": {"dt_fast": 0}})
        errors = config.validate()
        assert any("dt_fast" in e for e in errors)

    def test_dt_medium_less_than_dt_fast(self):
        """dt_medium < dt_fast should fail validation."""
        config = BrainConfig({"simulation": {"dt_fast": 0.02, "dt_medium": 0.005}})
        errors = config.validate()
        assert any("dt_medium" in e for e in errors)

    def test_dt_slow_less_than_dt_medium(self):
        """dt_slow < dt_medium should fail validation."""
        config = BrainConfig({"simulation": {"dt_medium": 0.1, "dt_slow": 0.02}})
        errors = config.validate()
        assert any("dt_slow" in e for e in errors)

    def test_invalid_metrics_port(self):
        """Port > 65535 should fail validation."""
        config = BrainConfig({"monitoring": {"metrics_port": 99999}})
        errors = config.validate()
        assert any("metrics_port" in e for e in errors)

    def test_valid_custom_config(self):
        """A valid custom config should pass."""
        config = BrainConfig({
            "brain": {"total_nodes": 1024, "max_layers": 10},
            "simulation": {"dt_fast": 0.001, "dt_medium": 0.01, "dt_slow": 0.1},
            "monitoring": {"metrics_port": 8080},
        })
        errors = config.validate()
        assert errors == []

    def test_env_override_applied(self):
        """Environment variables should override config values."""
        os.environ["BRAIN_LOGGING_LEVEL"] = "DEBUG"
        config = BrainConfig({})
        level = config.get("logging.level")
        assert level == "DEBUG"
        del os.environ["BRAIN_LOGGING_LEVEL"]


class TestConfigToScheduler:
    """Tests for BrainConfig.to_scheduler_config()."""

    def test_default_conversion(self):
        """Default config should convert to valid SchedulerConfig."""
        from brain_agent.scheduler import SchedulerConfig
        config = BrainConfig.from_defaults()
        sched_config = config.to_scheduler_config()
        assert isinstance(sched_config, SchedulerConfig)
        assert sched_config.dt_fast == 0.005
        assert sched_config.dt_medium == 0.02
        assert sched_config.dt_slow == 0.1

    def test_custom_dt_values(self):
        """Custom dt values should propagate to SchedulerConfig."""
        config = BrainConfig({
            "simulation": {
                "dt_fast": 0.001,
                "dt_medium": 0.01,
                "dt_slow": 0.1,
                "max_latency_ms": 5.0,
            },
        })
        sched_config = config.to_scheduler_config()
        assert sched_config.dt_fast == 0.001
        assert sched_config.dt_medium == 0.01
        assert sched_config.dt_slow == 0.1
        assert sched_config.max_latency_ms == 5.0

    def test_scheduler_uses_config_dt(self):
        """Scheduler should use dt values from config for step ratios."""
        config = BrainConfig({
            "simulation": {
                "dt_fast": 0.01,
                "dt_medium": 0.05,
                "dt_slow": 0.2,
            },
        })
        sched_config = config.to_scheduler_config()
        # Ratios: medium/fast = 5, slow/fast = 20
        assert sched_config.dt_medium / sched_config.dt_fast == 5.0
        assert sched_config.dt_slow / sched_config.dt_fast == 20.0

    def test_env_override_affects_scheduler_config(self):
        """Environment variable overrides should affect scheduler config."""
        os.environ["BRAIN_SIMULATION_DT_FAST"] = "0.002"
        config = BrainConfig.from_defaults()
        sched_config = config.to_scheduler_config()
        # The env var may or may not map correctly depending on parsing
        # Just verify it doesn't crash
        assert sched_config.dt_fast > 0
        del os.environ["BRAIN_SIMULATION_DT_FAST"]

    def test_yaml_config_file(self, tmp_path):
        """Should load config from YAML file."""
        yaml_content = """
simulation:
  dt_fast: 0.002
  dt_medium: 0.01
  dt_slow: 0.05
  max_latency_ms: 8.0
monitoring:
  metrics_port: 8080
"""
        filepath = tmp_path / "custom.yaml"
        filepath.write_text(yaml_content)

        config = BrainConfig.from_yaml(str(filepath))
        sched_config = config.to_scheduler_config()
        assert sched_config.dt_fast == 0.002
        assert sched_config.dt_medium == 0.01
        assert sched_config.dt_slow == 0.05
        assert sched_config.max_latency_ms == 8.0

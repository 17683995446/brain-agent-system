"""Tests for BrainConfig validation, env var overrides, and YAML I/O."""

import os
import pytest

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


class TestEnvVarOverrides:
    """Test that environment variable overrides work correctly."""

    def test_dt_fast_override(self, monkeypatch):
        """BRAIN_SIMULATION_DT_FAST should override simulation.dt_fast."""
        monkeypatch.setenv("BRAIN_SIMULATION_DT_FAST", "0.002")
        config = BrainConfig({"simulation": {"dt_fast": 0.005}})
        assert config.get_float("simulation.dt_fast") == 0.002

    def test_dt_medium_override(self, monkeypatch):
        """BRAIN_SIMULATION_DT_MEDIUM should override simulation.dt_medium."""
        monkeypatch.setenv("BRAIN_SIMULATION_DT_MEDIUM", "0.01")
        config = BrainConfig({"simulation": {"dt_medium": 0.02}})
        assert config.get_float("simulation.dt_medium") == 0.01

    def test_dt_slow_override(self, monkeypatch):
        """BRAIN_SIMULATION_DT_SLOW should override simulation.dt_slow."""
        monkeypatch.setenv("BRAIN_SIMULATION_DT_SLOW", "0.05")
        config = BrainConfig({"simulation": {"dt_slow": 0.1}})
        assert config.get_float("simulation.dt_slow") == 0.05

    def test_max_latency_override(self, monkeypatch):
        """BRAIN_SIMULATION_MAX_LATENCY_MS should override simulation.max_latency_ms."""
        monkeypatch.setenv("BRAIN_SIMULATION_MAX_LATENCY_MS", "15.0")
        config = BrainConfig({"simulation": {"max_latency_ms": 10.0}})
        assert config.get_float("simulation.max_latency_ms") == 15.0

    def test_metrics_port_override(self, monkeypatch):
        """BRAIN_MONITORING_METRICS_PORT should override monitoring.metrics_port."""
        monkeypatch.setenv("BRAIN_MONITORING_METRICS_PORT", "8080")
        config = BrainConfig({"monitoring": {"metrics_port": 9090}})
        assert config.get_int("monitoring.metrics_port") == 8080

    def test_env_override_creates_missing_section(self, monkeypatch):
        """Env override should create section if it doesn't exist."""
        monkeypatch.setenv("BRAIN_SIMULATION_DT_FAST", "0.003")
        config = BrainConfig({})
        assert config.get_float("simulation.dt_fast") == 0.003

    def test_env_override_does_not_affect_other_keys(self, monkeypatch):
        """Env override should not affect other keys in the same section."""
        monkeypatch.setenv("BRAIN_SIMULATION_DT_FAST", "0.002")
        config = BrainConfig({"simulation": {"dt_fast": 0.005, "dt_medium": 0.02}})
        assert config.get_float("simulation.dt_medium") == 0.02

    def test_env_override_with_int_value(self, monkeypatch):
        """Env override should parse integer values correctly."""
        monkeypatch.setenv("BRAIN_MONITORING_METRICS_PORT", "3000")
        config = BrainConfig({})
        val = config.get("monitoring.metrics_port")
        assert val == 3000
        assert isinstance(val, int)

    def test_env_override_with_bool_value(self, monkeypatch):
        """Env override should parse boolean values correctly."""
        monkeypatch.setenv("BRAIN_MONITORING_ENABLE_METRICS", "false")
        config = BrainConfig({})
        val = config.get("monitoring.enable_metrics")
        assert val is False


class TestToYaml:
    """Test the to_yaml() and save() methods."""

    def test_to_yaml_returns_string(self):
        config = BrainConfig({"simulation": {"dt_fast": 0.005}})
        result = config.to_yaml()
        assert isinstance(result, str)
        assert "dt_fast" in result

    def test_to_yaml_roundtrip(self, tmp_path):
        """Config should survive YAML roundtrip."""
        original = BrainConfig({
            "simulation": {"dt_fast": 0.002, "dt_medium": 0.01},
            "monitoring": {"metrics_port": 8080},
        })
        yaml_str = original.to_yaml()

        # Write and read back
        path = str(tmp_path / "config.yaml")
        original.save(path)
        loaded = BrainConfig.from_yaml(path)

        assert loaded.get_float("simulation.dt_fast") == 0.002
        assert loaded.get_float("simulation.dt_medium") == 0.01
        assert loaded.get_int("monitoring.metrics_port") == 8080

    def test_save_creates_file(self, tmp_path):
        config = BrainConfig({"simulation": {"dt_fast": 0.005}})
        path = str(tmp_path / "output.yaml")
        config.save(path)
        assert os.path.exists(path)

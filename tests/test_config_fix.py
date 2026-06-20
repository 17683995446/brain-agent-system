"""Tests for configuration system fixes (Stage 3 usability)."""

import os
import tempfile
import pytest

from brain_agent.config import BrainConfig


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
    """Test the to_yaml() method."""

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


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config_no_errors(self):
        config = BrainConfig({
            "simulation": {
                "dt_fast": 0.005,
                "dt_medium": 0.02,
                "dt_slow": 0.1,
                "max_latency_ms": 10.0,
            },
            "monitoring": {"metrics_port": 9090},
        })
        errors = config.validate()
        assert len(errors) == 0

    def test_invalid_dt_fast_returns_error(self):
        config = BrainConfig({"simulation": {"dt_fast": -1.0}})
        errors = config.validate()
        assert any("dt_fast" in e for e in errors)

    def test_invalid_port_returns_error(self):
        config = BrainConfig({"monitoring": {"metrics_port": 99999}})
        errors = config.validate()
        assert any("metrics_port" in e for e in errors)

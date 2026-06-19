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

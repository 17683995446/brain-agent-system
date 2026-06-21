"""Configuration management for brain simulation.

Supports layered configuration: YAML file -> environment variables -> code.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import yaml

from .constants import (
    DEFAULT_DT_FAST, DEFAULT_DT_MEDIUM, DEFAULT_DT_SLOW,
    DEFAULT_MAX_LATENCY_MS, DEFAULT_METRICS_PORT,
)

if TYPE_CHECKING:
    from .scheduler import SchedulerConfig


class BrainConfig:
    """Brain simulation configuration container."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = config_dict or {}
        self._apply_env_overrides()

    @classmethod
    def from_yaml(cls, path: str) -> "BrainConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data)

    @classmethod
    def from_defaults(cls) -> "BrainConfig":
        """Load default configuration."""
        default_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        if default_path.exists():
            return cls.from_yaml(str(default_path))
        return cls({})

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides.

        Environment variables use the prefix BRAIN_ followed by the
        dot-separated config path with dots replaced by underscores.
        For example, BRAIN_SIMULATION_DT_FAST maps to simulation.dt_fast.

        The first underscore separates the top-level section (e.g., "simulation")
        from the rest. The remaining part is used as-is (not split further),
        preserving underscores in keys like "dt_fast".
        """
        prefix = "BRAIN_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            config_key = key[len(prefix):].lower()
            # Split only on the first underscore to get the section name.
            # The rest is the key within that section (may contain underscores).
            if "_" not in config_key:
                # Single-word key, e.g., BRAIN_LOGLEVEL -> loglevel
                self._config[config_key] = self._parse_env_value(value)
                continue
            section, _, sub_key = config_key.partition("_")
            if section not in self._config or not isinstance(self._config[section], dict):
                self._config[section] = {}
            self._config[section][sub_key] = self._parse_env_value(value)

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        parts = key.split(".")
        current = self._config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        value = self.get(key, default)
        return int(value) if value is not None else default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        value = self.get(key, default)
        return float(value) if value is not None else default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1")
        return bool(value)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()

    def to_yaml(self) -> str:
        """Serialize configuration to YAML string.

        Returns:
            YAML-formatted string of the configuration.
        """
        result: str = yaml.dump(self._config, default_flow_style=False, sort_keys=True)
        return result

    def save(self, path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    def validate(self) -> list:
        """Validate configuration values.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []

        # Validate brain section
        total_nodes = self.get_int("brain.total_nodes", 2047)
        if total_nodes <= 0:
            errors.append(f"brain.total_nodes must be positive, got {total_nodes}")

        max_layers = self.get_int("brain.max_layers", 11)
        if max_layers <= 0 or max_layers > 20:
            errors.append(f"brain.max_layers must be 1-20, got {max_layers}")

        # Validate simulation section
        dt_fast = self.get_float("simulation.dt_fast", DEFAULT_DT_FAST)
        if dt_fast <= 0 or dt_fast > 1.0:
            errors.append(f"simulation.dt_fast must be (0, 1.0], got {dt_fast}")

        dt_medium = self.get_float("simulation.dt_medium", DEFAULT_DT_MEDIUM)
        if dt_medium <= 0 or dt_medium < dt_fast:
            errors.append(
                f"simulation.dt_medium must be >= dt_fast ({dt_fast}), got {dt_medium}"
            )

        dt_slow = self.get_float("simulation.dt_slow", DEFAULT_DT_SLOW)
        if dt_slow <= 0 or dt_slow < dt_medium:
            errors.append(
                f"simulation.dt_slow must be >= dt_medium ({dt_medium}), got {dt_slow}"
            )

        max_latency = self.get_float("simulation.max_latency_ms", DEFAULT_MAX_LATENCY_MS)
        if max_latency <= 0:
            errors.append(f"simulation.max_latency_ms must be positive, got {max_latency}")

        # Validate monitoring section
        metrics_port = self.get_int("monitoring.metrics_port", DEFAULT_METRICS_PORT)
        if metrics_port < 1 or metrics_port > 65535:
            errors.append(f"monitoring.metrics_port must be 1-65535, got {metrics_port}")

        return errors

    def to_scheduler_config(self) -> "SchedulerConfig":
        """Convert BrainConfig to SchedulerConfig.

        Maps the YAML/environment configuration to the scheduler's
        dataclass configuration. This bridges the config system to
        the runtime scheduler.

        Returns:
            SchedulerConfig instance populated from this BrainConfig.
        """
        from .scheduler import SchedulerConfig

        return SchedulerConfig(
            dt_fast=self.get_float("simulation.dt_fast", DEFAULT_DT_FAST),
            dt_medium=self.get_float("simulation.dt_medium", DEFAULT_DT_MEDIUM),
            dt_slow=self.get_float("simulation.dt_slow", DEFAULT_DT_SLOW),
            max_latency_ms=self.get_float("simulation.max_latency_ms", DEFAULT_MAX_LATENCY_MS),
            max_steps=self.get_int("simulation.max_steps", 0),
            enable_metrics=self.get_bool("monitoring.enable_metrics", True),
        )

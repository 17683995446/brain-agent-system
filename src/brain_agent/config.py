"""Configuration management for brain simulation.

Supports layered configuration: YAML file -> environment variables -> code.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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
        """Apply environment variable overrides."""
        prefix = "BRAIN_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                parts = config_key.split("_")
                current = self._config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = self._parse_env_value(value)

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

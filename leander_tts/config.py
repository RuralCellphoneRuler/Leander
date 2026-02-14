"""Configuration utilities using PyYAML (OmegaConf-compatible interface).

Provides a simple DictConfig-like wrapper so the codebase can work
with either OmegaConf or plain YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DictConfig:
    """Simple attribute-access dict wrapper, mimicking OmegaConf.DictConfig."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictConfig(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    DictConfig(v) if isinstance(v, dict) else v for v in value
                ])
            else:
                setattr(self, key, value)
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and key != "_data"

    def __repr__(self) -> str:
        return f"DictConfig({self._data})"

    def to_dict(self) -> dict:
        return self._data


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file and return a DictConfig."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return DictConfig(data)

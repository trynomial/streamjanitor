"""Standard locations (XDG), so an installed streamjanitor needs no project checkout.

    config:       $XDG_CONFIG_HOME/streamjanitor/config.toml   (~/.config/streamjanitor/)
    live library: $XDG_DATA_HOME/streamjanitor/library          (~/.local/share/streamjanitor/)
    studio:       $XDG_DATA_HOME/streamjanitor/studio
"""

import os
from importlib import resources
from pathlib import Path

APP = "streamjanitor"


def _xdg(var: str, fallback: str) -> Path:
    value = os.environ.get(var)
    return Path(value) if value and Path(value).is_absolute() else Path.home() / fallback


def config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / APP


def data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / APP


def default_config() -> Path:
    return config_dir() / "config.toml"


def default_library() -> Path:
    return data_dir() / "library"


def default_studio() -> Path:
    return data_dir() / "studio"


def example_config() -> str:
    return resources.files(APP).joinpath("config.example.toml").read_text(encoding="utf-8")

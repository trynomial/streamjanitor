"""TOML configuration. Relative paths are resolved against the config file's directory.

Targets come from `[[targets]]` blocks and/or from a library folder (`library = "..."`),
the format the web Studio produces.
"""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .detector import (
    DEFAULT_ENERGY_GATE,
    DEFAULT_MIN_VOICED,
    DEFAULT_PEAK_HOLD_FRAMES,
    DEFAULT_TAU,
)
from .features import HOP_S

DEFAULT_SAMPLE_RATE = 96000
SAMPLE_RATES = (96000, 48000)  # multiples of the 16 kHz analysis rate
MODES = ("replace", "report", "passthrough")  # see pipeline.py


@dataclass
class AudioConfig:
    input_device: str = "plughw:CARD=Loopback,DEV=1"
    output_device: str = "default"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 2
    delay_s: float = 4.0
    period_size: int = 1024
    alsa_latency_ms: int = 100
    crossfade_ms: float = 20.0


@dataclass
class DetectorConfig:
    energy_gate: float = DEFAULT_ENERGY_GATE
    min_voiced: float = DEFAULT_MIN_VOICED
    peak_hold_ms: float = DEFAULT_PEAK_HOLD_FRAMES * HOP_S * 1000

    @property
    def peak_hold_frames(self) -> int:
        return max(1, round(self.peak_hold_ms / 1000 / HOP_S))


@dataclass
class TargetConfig:
    file: Path
    name: str | None = None
    replace: Path | None = None
    tau: float = DEFAULT_TAU
    duration_s: float | None = None
    enabled: bool = True


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    targets: list[TargetConfig] = field(default_factory=list)
    library: Path | None = None


def _build(cls, table: dict, where: str):
    known = {f.name for f in fields(cls)}
    unknown = set(table) - known
    if unknown:
        raise ValueError(f"[{where}]: unknown keys {sorted(unknown)} (valid: {sorted(known)})")
    return cls(**table)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    unknown = set(data) - {"audio", "detector", "targets", "library"}
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")

    base = path.parent
    targets = []
    for i, item in enumerate(data.get("targets", [])):
        target = _build(TargetConfig, item, f"targets[{i}]")
        target.file = base / target.file
        target.replace = base / target.replace if target.replace else None  # "" = silence
        targets.append(target)

    audio = _build(AudioConfig, data.get("audio", {}), "audio")
    library = base / data["library"] if "library" in data else None
    if library is not None:
        targets += library_targets(library, audio.sample_rate)

    return Config(
        audio=audio,
        detector=_build(DetectorConfig, data.get("detector", {}), "detector"),
        targets=targets,
        library=library,
    )


def library_targets(root: Path, sample_rate: int) -> list[TargetConfig]:
    """Targets of a library folder; none if it doesn't exist yet (e.g. before the first import)."""
    from .library import LIBRARY_FILE, Library

    if not (root / LIBRARY_FILE).exists():
        return []
    lib = Library.load(root)
    if lib.pairs and lib.sample_rate != sample_rate:
        raise ValueError(f"library {root} is for {lib.sample_rate} Hz but audio.sample_rate is {sample_rate}")
    return [
        TargetConfig(
            file=lib.path(p.target),
            name=p.name,
            replace=lib.path(p.replacement) if p.replacement else None,
            tau=p.tau,
            enabled=p.enabled,
        )
        for p in lib.pairs
    ]

"""Calibration: scan a long recording (e.g. an archived episode) with a template.

Finds where the target airs, how well it matches there, the best match anywhere
else, and suggests a threshold tau between the two. Shared by the CLI and the
web Studio.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audiofile import decode_chunks
from .detector import DEFAULT_ENERGY_GATE, DEFAULT_MIN_VOICED, HeadTemplate, score_trace
from .features import HOP_S, NUM_MELS, Decimator, LogMelExtractor

MIN_BACKGROUND_S = 60.0  # audio needed besides the target to judge false matches


@dataclass
class Match:
    start_s: float  # where the target would start in the recording
    score: float
    is_target: bool


@dataclass
class Calibration:
    matches: list[Match]  # best distinct matches, best first
    occurrences: int  # how many of them are the target (it may air more than once)
    weakest: float  # lowest score among the occurrences
    background: float  # best score anywhere else
    suggested_tau: float | None  # None when the recording is too short to judge false matches
    notes: list[str] = field(default_factory=list)

    @property
    def margin(self) -> float:
        return self.weakest - self.background


def recording_features(path: str | Path, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Features of a whole recording, decoded in chunks (hours of audio fit in little memory)."""
    decimator, extractor = Decimator(sample_rate), LogMelExtractor()
    frames, rms = [], []
    for chunk in decode_chunks(path, sample_rate, 1, sample_rate * 10):
        f, r = extractor.process(decimator.process(chunk[:, 0]))
        frames.append(f)
        rms.append(r)
    if not frames:
        return np.zeros((0, NUM_MELS), np.float32), np.zeros(0, np.float32)
    return np.concatenate(frames), np.concatenate(rms)


def calibrate(
    template: HeadTemplate,
    frames: np.ndarray,
    rms: np.ndarray,
    gate: float = DEFAULT_ENERGY_GATE,
    min_voiced: float = DEFAULT_MIN_VOICED,
    top: int = 8,
) -> Calibration:
    """Raises ValueError if nothing in the recording can be scored (e.g. it is silent or too short)."""
    width, offset = template.width, template.head_offset
    trace = score_trace(template.head, frames, rms, gate, min_voiced)

    # Distinct peaks, best first.
    remaining = trace.copy()
    peaks = []
    for _ in range(max(top, 30)):
        k = int(np.argmax(remaining)) if len(remaining) else 0
        if not len(remaining) or remaining[k] <= 0:
            break
        peaks.append((float(trace[k]), k))
        remaining[max(0, k - width) : k + width] = 0
    if not peaks:
        raise ValueError("no match at all: is the recording silent or shorter than the head?")

    # Occurrences of the target are the peaks above the largest score gap; the rest is background.
    scores = [s for s, _ in peaks]
    n_occ = int(np.argmax(-np.diff(scores))) + 1 if len(scores) > 1 else 1

    # While a target plays the detector is muted, so matches inside an occurrence don't count.
    outside = np.ones(len(trace), dtype=bool)
    for _, k in peaks[:n_occ]:
        outside[max(0, k - width) : k + template.duration_frames] = False
    background = float(trace[outside].max()) if outside.any() else 0.0
    weakest = min(scores[:n_occ])
    margin = weakest - background

    notes = []
    background_s = float(np.count_nonzero(outside)) * HOP_S
    suggested = round(background + 0.4 * margin, 2)
    if background_s < MIN_BACKGROUND_S:
        suggested = None
        notes.append(f"only {background_s:.0f} s of audio besides the target: too little to judge false "
                     f"matches, so no τ is suggested. Use a whole episode.")
    if weakest < 0.7:
        notes.append("no convincing match: is the target really in this recording?")
    elif weakest > 0.97:
        notes.append("a ~1.0 match usually means the template was cut from this very recording; the live "
                     "stream (another encoding) will score lower. If you can, check on another recording.")
    if suggested is not None and margin < 0.15:
        notes.append("small margin: try a longer head or a more distinctive head start.")

    matches = [
        Match(round((k - (width - 1) - offset) * HOP_S, 2), round(s, 3), i < n_occ)
        for i, (s, k) in enumerate(peaks[:top])
    ]
    return Calibration(matches, n_occ, round(weakest, 3), round(background, 3), suggested, notes)

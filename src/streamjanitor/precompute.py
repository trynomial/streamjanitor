"""Target audio -> HeadTemplate (shared by the CLI and the web Studio)."""

import numpy as np

from .detector import DEFAULT_ENERGY_GATE, DEFAULT_PEAK_HOLD_FRAMES, HeadTemplate
from .features import FEATURE_RATE, HOP, HOP_S, WINDOW, extract_features

DEFAULT_HEAD_S = 3.0
MIN_HEAD_FRAMES = 25  # 0.5 s


def make_template(
    audio: np.ndarray,
    sample_rate: int,
    target_id: str,
    name: str,
    head_s: float = DEFAULT_HEAD_S,
    head_offset_s: float | None = None,
    gate: float = DEFAULT_ENERGY_GATE,
) -> tuple[HeadTemplate, list[str]]:
    """Mono target audio at the pipeline rate -> (template, warnings).

    head_offset_s=None skips leading silence. Raises ValueError if no usable head exists.
    """
    duration_s = len(audio) / sample_rate
    frames, rms = extract_features(audio, sample_rate)
    warnings = []

    if head_offset_s is None:
        voiced = np.flatnonzero(rms >= gate)
        if not len(voiced):
            raise ValueError("the target is silent")
        offset = int(voiced[0])
    else:
        offset = round(head_offset_s / HOP_S)

    width = round(head_s / HOP_S)
    if offset + width > len(frames):
        width = len(frames) - offset
        warnings.append(f"target too short: head reduced to {width * HOP_S:.2f} s")
    if width < MIN_HEAD_FRAMES:
        raise ValueError("head shorter than 0.5 s: check the head offset and the target audio")

    voiced_fraction = float(np.mean(rms[offset : offset + width] >= gate))
    if voiced_fraction < 0.9:
        warnings.append(f"only {voiced_fraction:.0%} of the head is above the silence gate; "
                        "consider a later head offset")

    template = HeadTemplate(target_id, name, frames[offset : offset + width], offset, duration_s)
    return template, warnings


def detection_latency_s(template: HeadTemplate, peak_hold_frames: int = DEFAULT_PEAK_HOLD_FRAMES) -> float:
    """Seconds from the target start until it is recognised (excluding audio buffering)."""
    return ((template.head_offset + template.width + peak_hold_frames) * HOP + WINDOW) / FEATURE_RATE

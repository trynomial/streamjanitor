"""Target detection by matching only the opening seconds ("head") of each target.

A template stores W feature frames taken `head_offset` frames after the start of
the target, plus the target's full duration. The live detector keeps the last W
frames and scores them against each head with the mean per-frame cosine
similarity. The score peaks sharply (a frame or two wide) when the stream lines
up with the head; the peak position gives the exact frame where the target
started, and start + duration gives where it ends.

Cost per target and per 20 ms frame: one W x 60 dot product.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import HOP_S, NUM_MELS

TEMPLATE_VERSION = 2

DEFAULT_TAU = 0.6  # score a window must reach to count as the target's head
DEFAULT_ENERGY_GATE = 0.003  # frame RMS below this counts as silence
DEFAULT_MIN_VOICED = 0.5  # fraction of non-silent frames a window needs to be scored
DEFAULT_PEAK_HOLD_FRAMES = 5  # a peak is final once no better score came for this long (100 ms)


def unit_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row; all-zero rows (digital silence) stay zero."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-9)


@dataclass
class HeadTemplate:
    target_id: str
    name: str
    head: np.ndarray  # (W, NUM_MELS)
    head_offset: int  # frames from target start to the first head frame
    duration_s: float  # full target duration, measured from the target start
    tau: float = DEFAULT_TAU

    def __post_init__(self) -> None:
        self.head = unit_rows(self.head)
        if self.head.ndim != 2 or self.head.shape[1] != NUM_MELS or len(self.head) == 0:
            raise ValueError(f"template head must have shape (W, {NUM_MELS}), got {self.head.shape}")

    @property
    def width(self) -> int:
        return len(self.head)

    @property
    def duration_frames(self) -> int:
        return round(self.duration_s / HOP_S)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            version=np.int32(TEMPLATE_VERSION),
            head=self.head,
            head_offset=np.int32(self.head_offset),
            duration_s=np.float64(self.duration_s),
            target_id=np.str_(self.target_id),
            name=np.str_(self.name),
        )

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "HeadTemplate":
        with np.load(path, allow_pickle=False) as data:
            if "version" not in data or int(data["version"]) != TEMPLATE_VERSION:
                raise ValueError(f"{path}: unsupported template format, re-run `streamjanitor precompute`")
            kwargs = {
                "target_id": str(data["target_id"]),
                "name": str(data["name"]),
                "head": data["head"],
                "head_offset": int(data["head_offset"]),
                "duration_s": float(data["duration_s"]),
            }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)


@dataclass(frozen=True)
class Detection:
    target_id: str
    name: str
    score: float
    peak_frame: int  # live frame index where the head's last frame lined up
    start_frame: int  # live frame index where the target started


def score_trace(
    head: np.ndarray,
    frames: np.ndarray,
    rms: np.ndarray,
    energy_gate: float,
    min_voiced: float,
) -> np.ndarray:
    """Offline equivalent of HeadDetector scoring over a whole recording.

    out[k] is the score of the window ending at frame k (0 where k < W - 1 or
    the window is mostly below the energy gate).
    """
    head = unit_rows(head)
    width, n = len(head), len(frames)
    out = np.zeros(n, dtype=np.float32)
    if n < width:
        return out

    n_scores = n - width + 1
    scores = np.empty(n_scores, dtype=np.float64)
    block = 20000  # windows per block: bounds memory to block x W cosines on long recordings
    for b in range(0, n_scores, block):
        m = min(block, n_scores - b)
        per_frame = unit_rows(frames[b : b + m + width - 1]) @ head.T  # cosine of each frame vs each head frame
        acc = np.zeros(m, dtype=np.float64)
        for w in range(width):
            acc += per_frame[w : w + m, w]
        scores[b : b + m] = acc / width

    voiced = np.concatenate(([0], np.cumsum(rms >= energy_gate)))
    voiced_fraction = (voiced[width:] - voiced[:-width]) / width
    scores[voiced_fraction < min_voiced] = 0.0
    out[width - 1 :] = scores
    return out


class _Group:
    """Targets sharing a head width, scored together with one matrix-vector product per frame.

    Per-target state lives in arrays, so a frame costs the same few numpy calls whether the
    group holds one target or fifty.
    """

    def __init__(self, targets: list[HeadTemplate], previous: dict[str, tuple[float, int, int]]) -> None:
        self.targets = targets
        self.width = targets[0].width
        self.heads = np.stack([t.head.reshape(-1) for t in targets])  # (T, W * NUM_MELS)
        self.tau = np.array([t.tau for t in targets])
        self.duration = np.array([t.duration_frames for t in targets])
        self.scores = np.zeros(len(targets))
        # Peak candidates (best score, frame; -inf = none) and muting carry over across reloads.
        state = [previous.get(t.target_id, (-np.inf, 0, 0)) for t in targets]
        self.best = np.array([b for b, _, _ in state])
        self.best_frame = np.array([f for _, f, _ in state])
        self.muted_until = np.array([m for _, _, m in state])  # first frame after the target ends

    def state(self) -> dict[str, tuple[float, int, int]]:
        return {t.target_id: (self.best[i], self.best_frame[i], self.muted_until[i])
                for i, t in enumerate(self.targets)}


class HeadDetector:
    """Streaming detector: feed feature frames, get Detections back."""

    def __init__(
        self,
        targets: list[HeadTemplate],
        energy_gate: float = DEFAULT_ENERGY_GATE,
        min_voiced: float = DEFAULT_MIN_VOICED,
        peak_hold_frames: int = DEFAULT_PEAK_HOLD_FRAMES,
    ) -> None:
        self.energy_gate = energy_gate
        self.min_voiced = min_voiced
        self.peak_hold_frames = peak_hold_frames
        self.frame_index = -1  # index of the newest frame
        self._max_w = 1
        self._frames = np.zeros((2, NUM_MELS), dtype=np.float32)
        self._voiced = np.zeros(2, dtype=bool)
        self._slot = 0
        self._groups: list[_Group] = []
        self.set_targets(targets)

    @property
    def targets(self) -> list[HeadTemplate]:
        return [t for g in self._groups for t in g.targets]

    @property
    def last_scores(self) -> dict[str, float]:
        """Score of each target on the newest frame (for tests and debugging)."""
        return {t.target_id: float(g.scores[i]) for g in self._groups for i, t in enumerate(g.targets)}

    def set_targets(self, targets: list[HeadTemplate]) -> None:
        """Swap the target set mid-stream, keeping the frame count, the recent frames, and the
        peak/muting state of targets that stay."""
        previous = {tid: st for g in self._groups for tid, st in g.state().items()}
        by_width: dict[int, list[HeadTemplate]] = {}
        for t in targets:
            by_width.setdefault(t.width, []).append(t)
        self._groups = [_Group(ts, previous) for ts in by_width.values()]

        # Doubled circular buffer: the last max_w frames are always one contiguous slice.
        old_max = self._max_w
        recent = self._frames[self._slot : self._slot + old_max].copy()
        recent_voiced = self._voiced[self._slot : self._slot + old_max].copy()
        self._max_w = max(by_width, default=1)
        keep = min(old_max, self._max_w)
        self._frames = np.zeros((2 * self._max_w, NUM_MELS), dtype=np.float32)
        self._voiced = np.zeros(2 * self._max_w, dtype=bool)
        self._frames[:keep] = self._frames[self._max_w : self._max_w + keep] = recent[old_max - keep :]
        self._voiced[:keep] = self._voiced[self._max_w : self._max_w + keep] = recent_voiced[old_max - keep :]
        self._slot = keep % self._max_w

    def process(self, frames: np.ndarray, rms: np.ndarray) -> list[Detection]:
        detections = []
        for frame, frame_rms in zip(unit_rows(frames), rms, strict=True):
            self._push(frame, bool(frame_rms >= self.energy_gate))
            for group in self._groups:
                detections += self._update(group)
        return detections

    def _push(self, frame: np.ndarray, voiced: bool) -> None:
        self.frame_index += 1
        i = self._slot
        self._frames[i] = self._frames[i + self._max_w] = frame
        self._voiced[i] = self._voiced[i + self._max_w] = voiced
        self._slot = (i + 1) % self._max_w

    def _update(self, g: _Group) -> list[Detection]:
        k, w = self.frame_index, g.width
        if k + 1 < w:
            return []
        end = self._slot + self._max_w
        if self._voiced[end - w : end].mean() < self.min_voiced:
            g.scores[:] = 0.0
        else:  # mean per-frame cosine of the last w frames with every head at once
            g.scores = g.heads @ self._frames[end - w : end].reshape(-1) / w

        live = k >= g.muted_until
        better = live & (g.scores >= g.tau) & (g.scores > g.best)
        g.best[better] = g.scores[better]
        g.best_frame[better] = k
        # A candidate becomes the peak when no better score came for peak_hold_frames.
        final = np.flatnonzero((g.best > -np.inf) & (k - g.best_frame >= self.peak_hold_frames))
        detections = []
        for i in final:
            t, peak = g.targets[i], int(g.best_frame[i])
            start = peak - (w - 1) - t.head_offset
            detections.append(Detection(t.target_id, t.name, float(g.best[i]), peak, start))
            g.best[i] = -np.inf
            # A song may repeat its opening later on: stay quiet until the target is over.
            g.muted_until[i] = start + g.duration[i]
        return detections

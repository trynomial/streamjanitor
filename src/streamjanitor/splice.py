"""Sample-accurate splicing on the delayed output.

A Cut covers absolute input samples [start, end). Inside it the live stream is
replaced by the clip (or silence); at both edges live and clip are
equal-power crossfaded over `fade` samples, so neither side is ever cut hard.
"""

import threading
from dataclasses import dataclass

import numpy as np

from .audio_io import AudioRingBuffer
from .audiofile import Clip


@dataclass
class Cut:
    start: int
    end: int
    clip: Clip | None  # played from `start`


class SpliceEngine:
    def __init__(self, ring: AudioRingBuffer, channels: int, fade_frames: int) -> None:
        self.ring = ring
        self.channels = channels
        self.fade = max(1, fade_frames)
        self._cuts: list[Cut] = []
        self._lock = threading.Lock()

    def schedule(self, start: int, end: int, clip: Clip | None) -> int:
        """Queue a cut; returns how many target samples were already played (leak)."""
        now = self.ring.read_position
        leak = max(0, min(now, end) - start)
        start = max(start, now)
        if end <= start:
            return leak
        with self._lock:
            for cut in self._cuts:
                if start < cut.end and cut.start < end:
                    cut.end = max(cut.end, end)  # overlapping targets: keep the first clip, extend
                    return leak
            self._cuts.append(Cut(start, end, clip))
        if clip is not None:
            clip.prefetch()  # it plays in a few seconds: get it off the disk now
        return leak

    def cancel_all(self) -> None:
        """Drop future cuts; a cut already playing fades back to the live stream."""
        now = self.ring.read_position
        with self._lock:
            kept = []
            for cut in self._cuts:
                if cut.start < now:
                    cut.end = min(cut.end, now + self.fade)
                    kept.append(cut)
            self._cuts = kept

    @property
    def pending(self) -> list[Cut]:
        with self._lock:
            return list(self._cuts)

    def render(self, n: int) -> tuple[np.ndarray, int | None]:
        live, pos = self.ring.read(n)
        if pos is None:
            return live, None
        with self._lock:
            self._cuts = [c for c in self._cuts if c.end > pos]
            active = [c for c in self._cuts if c.start < pos + n]
        if not active:
            return live, pos

        t = pos + np.arange(n)
        out = live
        for cut in active:
            ramp = np.clip(np.minimum(t - cut.start, cut.end - t) / self.fade, 0.0, 1.0)
            g_clip = np.sin(0.5 * np.pi * ramp)[:, None]
            g_live = np.where(ramp < 1.0, np.cos(0.5 * np.pi * ramp), 0.0)[:, None]
            clip_frames = np.zeros((n, self.channels))
            if cut.clip is not None:
                offset = pos - cut.start  # clip frame played at output position pos
                a, b = max(0, -offset), min(n, len(cut.clip) - offset)
                if a < b:
                    clip_frames[a:b] = cut.clip.frames(offset + a, b - a)
            out = out * g_live + clip_frames * g_clip
        return np.clip(out, -1.0, 1.0), pos

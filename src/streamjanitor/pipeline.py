"""The processing chain, independent of where audio comes from.

    capture(frames)  : input -> delay line
    analyze(frames)  : the same input -> mono 16 kHz -> features -> head detector
                       -> cuts scheduled on the delay line
    render(n)        : delayed output with cuts applied

The daemon runs the three stages in separate threads (analysis in blocks of
ANALYSIS_BLOCK_S: fewer, larger numpy calls cost far less on a Pi); `simulate`
and the tests call them in sequence on a file.

Modes:
    replace      recognised targets are cut and replaced
    report       recognition runs and detections are recorded, nothing is cut
    passthrough  no recognition at all: the output is the input, delayed, sample for sample
"""

import logging
import threading
import time
from collections import deque

import numpy as np

from .audio_io import AudioRingBuffer
from .audiofile import Clip, load_clip
from .config import MODES, AudioConfig, Config, DetectorConfig
from .detector import Detection, HeadDetector, HeadTemplate
from .features import HOP, HOP_S, Decimator, LogMelExtractor, decimation_factor
from .precompute import detection_latency_s
from .splice import SpliceEngine

logger = logging.getLogger(__name__)

ANALYSIS_BLOCK_S = 0.1


class Pipeline:
    def __init__(
        self,
        targets: list[HeadTemplate],
        clips: dict[str, Clip],
        audio: AudioConfig | None = None,
        detector: DetectorConfig | None = None,
    ) -> None:
        self.audio = audio = audio or AudioConfig()
        detector = detector or DetectorConfig()
        self.sample_rate = audio.sample_rate
        self.samples_per_frame = HOP * decimation_factor(audio.sample_rate)
        self.targets = {t.target_id: t for t in targets}
        self.clips = clips  # target_id -> replacement

        self.ring = AudioRingBuffer(audio.channels, audio.sample_rate, audio.delay_s)
        self.splice = SpliceEngine(self.ring, audio.channels, round(audio.crossfade_ms / 1000 * audio.sample_rate))
        self._detector_cfg = detector
        self._peak_hold = detector.peak_hold_frames
        self._lock = threading.Lock()  # targets and mode
        self.mode = "replace"
        self._analyzed = 0  # input samples seen by analyze()
        self._reset_analysis(targets)
        self.history: deque[dict] = deque(maxlen=50)
        self._warn_latency(targets)

    @classmethod
    def from_config(cls, cfg: Config) -> "Pipeline":
        return cls(*load_targets(cfg), cfg.audio, cfg.detector)

    def detection_latency_s(self, t: HeadTemplate) -> float:
        """From the target start to the cut being scheduled, device buffering and analysis blocks included."""
        buffering = 2 * self.audio.period_size / self.sample_rate + ANALYSIS_BLOCK_S
        return detection_latency_s(t, self._peak_hold) + buffering

    def _warn_latency(self, targets: list[HeadTemplate]) -> None:
        delay = self.audio.delay_s
        for t in targets:
            needed = self.detection_latency_s(t)
            if needed > delay:
                logger.warning(
                    "Target '%s' is recognised ~%.1f s after it starts but delay_s is %.1f: "
                    "about %.1f s of it will be heard. Raise delay_s or use a shorter head.",
                    t.name, needed, delay, needed - delay,
                )

    def _reset_analysis(self, targets: list[HeadTemplate]) -> None:
        """Fresh analysis chain whose frame 0 starts at the next sample analyze() gets."""
        d = self._detector_cfg
        self.decimator = Decimator(self.sample_rate)
        self.extractor = LogMelExtractor()
        self.detector = HeadDetector(targets, d.energy_gate, d.min_voiced, self._peak_hold)
        self._origin = self._analyzed  # absolute input sample of detector frame 0
        self._stale = False

    def set_targets(self, targets: list[HeadTemplate], clips: dict[str, Clip]) -> None:
        """Replace the targets without touching the audio path."""
        with self._lock:
            self.detector.set_targets(targets)
            self.targets = {t.target_id: t for t in targets}
            self.clips = clips
        self._warn_latency(targets)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r} (expected one of {', '.join(MODES)})")
        with self._lock:
            self.mode = mode
            if mode != "replace":
                self.splice.cancel_all()  # a cut already playing fades back to the live stream
        logger.info("Mode: %s", {
            "replace": "replacing targets",
            "report": "report only (targets are recognised, not cut)",
            "passthrough": "pass-through (no recognition, input copied to the output)",
        }[mode])

    def capture(self, frames: np.ndarray) -> None:
        self.ring.write(frames)

    def analyze(self, frames: np.ndarray) -> list[Detection]:
        """Captured frames, in the same order (any block size) -> detections, cuts scheduled."""
        if self.mode == "passthrough":
            self._stale = True  # skipped audio: restart the analysis from scratch afterwards
            self._analyzed += len(frames)
            return []
        if self._stale:
            with self._lock:
                self._reset_analysis(list(self.targets.values()))
        self._analyzed += len(frames)
        feats, rms = self.extractor.process(self.decimator.process(frames.mean(axis=1)))
        return self.detect(feats, rms)

    def detect(self, feats: np.ndarray, rms: np.ndarray) -> list[Detection]:
        with self._lock:
            if self.mode == "passthrough":  # switched while this block was being analysed
                return []
            detections = self.detector.process(feats, rms)
            for det in detections:
                self._handle(det)
        return detections

    def _handle(self, det: Detection) -> None:
        t = self.targets[det.target_id]
        rate = self.sample_rate
        first = self._origin + det.start_frame * self.samples_per_frame
        start = max(0, first)
        end = first + round(t.duration_s * rate)
        record = {"time": time.time(), "id": t.target_id, "name": t.name, "score": round(det.score, 3),
                  "leak_s": 0.0, "mode": self.mode}
        if self.mode == "report":
            logger.info("Detected '%s' (score %.3f): report only, not cutting", t.name, det.score)
        else:
            leak = self.splice.schedule(start, end, self.clips.get(t.target_id))
            record["leak_s"] = round(leak / rate, 2)
            logger.info(
                "Detected '%s' (score %.3f): cutting %.1f s from input t=%.2f s%s",
                t.name, det.score, (end - start) / rate, start / rate,
                f", {leak / rate:.2f} s already played" if leak else "",
            )
        self.history.append(record)

    def render(self, n: int) -> tuple[np.ndarray, int | None]:
        return self.splice.render(n)


def load_targets(cfg: Config) -> tuple[list[HeadTemplate], dict[str, Clip]]:
    """Enabled targets of a config and their replacement clips."""
    a = cfg.audio
    fade = round(a.crossfade_ms / 1000 * a.sample_rate)
    targets, clips, clip_cache = [], {}, {}
    for tc in cfg.targets:
        if not tc.enabled:
            continue
        t = HeadTemplate.load(tc.file, name=tc.name, tau=tc.tau, duration_s=tc.duration_s)
        if t.target_id in {x.target_id for x in targets}:
            raise ValueError(f"duplicate target id '{t.target_id}' ({tc.file})")
        targets.append(t)
        if tc.replace:
            if tc.replace not in clip_cache:
                clip_cache[tc.replace] = load_clip(tc.replace, a.sample_rate, a.channels, fade)
            clips[t.target_id] = clip_cache[tc.replace]
        logger.info(
            "Target '%s': head %.1f s at +%.1f s, duration %.1f s, tau %.2f, replace=%s",
            t.name, t.width * HOP_S, t.head_offset * HOP_S, t.duration_s, t.tau, tc.replace or "silence",
        )
    return targets, clips

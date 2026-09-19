"""Live daemon: ALSA loopback capture -> Pipeline -> ALSA output, in three threads.

- capture thread: blocking reads from the loopback, feeds the delay line and the
  analysis queue (never drops audio: frame indices must stay aligned with samples).
- analysis thread: features + detection (numpy releases the GIL in the heavy parts).
- playback thread: renders the delayed, spliced stream; paced by the DAC.

With `control=True` (used by the web On Air section) the daemon reads JSON commands from
stdin, one per line, and writes its status as a JSON line to stdout every second:
    {"cmd": "mode", "mode": "report"}   replace / report / passthrough (see pipeline.MODES)
    {"cmd": "reload"}                   re-read targets (config + library), audio untouched
    {"cmd": "stop"}
EOF on stdin stops the daemon, so it never outlives its supervisor.
"""

import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .audio_io import (
    SND_PCM_STREAM_CAPTURE,
    SND_PCM_STREAM_PLAYBACK,
    AlsaError,
    AlsaPcm,
)
from .config import load_config
from .pipeline import ANALYSIS_BLOCK_S, Pipeline, load_targets

logger = logging.getLogger(__name__)

STATS_INTERVAL_S = 600.0
PLAYBACK_START_TIMEOUT_S = 3.0
CAPTURE_RETRY_S = (0.5, 10.0)  # first and longest wait before reopening a failed capture device


class Daemon:
    def __init__(self, config_path: str | Path, control: bool = False) -> None:
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self.pipeline = Pipeline.from_config(self.cfg)
        self.control = control
        self.stop_event = threading.Event()
        self.started_at = time.time()
        self._analysis_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._capture: AlsaPcm | None = None
        self._playback: AlsaPcm | None = None
        self._crashes: list[BaseException] = []  # threads that died: the daemon exits with 1
        self._stdout_lock = threading.Lock()
        # Shown on the On Air page until the problem goes away.
        self.capture_errors = 0
        self._capture_error: str | None = None
        self._reload_error: str | None = None

    def start(self) -> None:
        a = self.cfg.audio
        # Playback must be running before capture starts: with snd-aloop's timer_source on
        # the DAC, the loopback only advances while the DAC plays, and reading it before
        # fails with an I/O error.
        self._playback = AlsaPcm(a.output_device, SND_PCM_STREAM_PLAYBACK, a.sample_rate, a.channels, a.alsa_latency_ms)
        loops = [("playback", self._playback_loop), ("analysis", self._analysis_loop)]
        if self.control:
            loops.append(("control", self._control_loop))
        for name, fn in loops:
            self._spawn(name, fn)

        started = a.alsa_latency_ms * a.sample_rate // 1000 + a.period_size  # device buffer full => running
        deadline = time.monotonic() + PLAYBACK_START_TIMEOUT_S
        while self._playback.frames_written < started and time.monotonic() < deadline and not self.stop_event.is_set():
            time.sleep(0.01)
        if self._playback.frames_written < started:
            logger.warning("%s did not start playing within %.0f s", a.output_device, PLAYBACK_START_TIMEOUT_S)

        self._spawn("capture", self._capture_loop)
        logger.info("Running: %s -> %s, %d Hz, delay %.1f s", a.input_device, a.output_device, a.sample_rate, a.delay_s)

    def _spawn(self, name: str, fn) -> None:
        t = threading.Thread(target=self._guard, args=(fn,), name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _guard(self, fn) -> None:
        try:
            fn()
        except BaseException as e:  # a dead thread must take the daemon down so it gets restarted
            logger.exception("Thread %s crashed", threading.current_thread().name)
            self._crashes.append(e)
            self.stop_event.set()

    def _capture_loop(self) -> None:
        """Read the loopback; on a device error, reopen it with backoff instead of dying,
        so playback (and with it the DAC clock the loopback may depend on) keeps running."""
        a = self.cfg.audio
        retry = CAPTURE_RETRY_S[0]
        failures = 0  # consecutive
        while not self.stop_event.is_set():
            try:
                if self._capture is None:
                    self._capture = AlsaPcm(a.input_device, SND_PCM_STREAM_CAPTURE, a.sample_rate, a.channels,
                                            a.alsa_latency_ms)
                frames = self._capture.read(a.period_size)
            except AlsaError as e:
                self.capture_errors += 1
                failures += 1
                self._capture_error = f"{time.strftime('%H:%M:%S')} capture: {e}"
                logger.error("Capture failed (%s); reopening in %.1f s", e, retry)
                if self._capture is not None:
                    self._capture.close()
                    self._capture = None
                self.stop_event.wait(retry)
                retry = min(retry * 2, CAPTURE_RETRY_S[1])
                continue
            if len(frames):
                if failures:
                    logger.info("Capture resumed after %d failed attempt(s)", failures)
                    self._capture_error, failures = None, 0
                retry = CAPTURE_RETRY_S[0]
                self.pipeline.capture(frames)
                self._analysis_queue.put(frames)

    def _analysis_loop(self) -> None:
        block = round(ANALYSIS_BLOCK_S * self.cfg.audio.sample_rate)
        pending: list[np.ndarray] = []
        while not self.stop_event.is_set():
            try:
                pending.append(self._analysis_queue.get(timeout=0.2))
            except queue.Empty:
                continue
            if sum(len(p) for p in pending) >= block:
                self.pipeline.analyze(np.concatenate(pending))
                pending = []

    def _playback_loop(self) -> None:
        period = self.cfg.audio.period_size
        while not self.stop_event.is_set():
            frames, _ = self.pipeline.render(period)
            self._playback.write(frames)

    def _control_loop(self) -> None:
        for line in sys.stdin:
            try:
                msg = json.loads(line)
                cmd = msg.get("cmd")
            except (json.JSONDecodeError, AttributeError):
                logger.warning("control: ignoring malformed line %r", line[:80])
                continue
            if cmd == "mode":
                try:
                    self.pipeline.set_mode(msg.get("mode"))
                except ValueError as e:
                    logger.warning("control: %s", e)
            elif cmd == "reload":
                self.reload()
            elif cmd == "stop":
                break
            else:
                logger.warning("control: unknown command %r", cmd)
            self._emit_status()
        self.stop_event.set()  # stdin closed or stop requested

    def reload(self) -> None:
        """Re-read the targets; audio settings changes need a restart."""
        try:
            cfg = load_config(self.config_path)
            targets, clips = load_targets(cfg)
        except Exception as e:
            self._reload_error = f"{time.strftime('%H:%M:%S')} reload failed: {e}"
            logger.error("Reload failed, keeping the current targets: %s", e)
            return
        if cfg.audio != self.cfg.audio or cfg.detector != self.cfg.detector:
            logger.warning("Audio/detector settings changed: they take effect after a restart")
        self.pipeline.set_targets(targets, clips)
        self.cfg.targets = cfg.targets
        self._reload_error = None
        logger.info("Reloaded: %d active target(s)", len(targets))

    def status(self) -> dict:
        ring = self.pipeline.ring
        rate = self.cfg.audio.sample_rate
        return {
            "pid": os.getpid(),
            "uptime_s": round(time.time() - self.started_at, 1),
            "mode": self.pipeline.mode,
            "active_targets": sorted(self.pipeline.targets),
            "delay_s": self.cfg.audio.delay_s,
            "fill_s": round(ring.fill / rate, 2),
            "analysis_backlog": self._analysis_queue.qsize(),
            "underruns": ring.underruns,
            "overruns": ring.overruns,
            "drift_corrections": ring.drift_corrections,
            "xruns_in": self._capture.xruns if self._capture else 0,
            "xruns_out": self._playback.xruns if self._playback else 0,
            "capture_errors": self.capture_errors,
            "history": list(self.pipeline.history)[-20:],
            "last_error": self._capture_error or self._reload_error,
        }

    def _emit_status(self) -> None:
        line = json.dumps({"status": self.status()})
        with self._stdout_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def run_forever(self) -> int:
        self.start()
        last_stats = time.monotonic()
        while not self.stop_event.wait(1.0):
            if self.control:
                self._emit_status()
            if time.monotonic() - last_stats >= STATS_INTERVAL_S:
                last_stats = time.monotonic()
                self._log_stats()
        self.shutdown()
        return 1 if self._crashes else 0

    def _log_stats(self) -> None:
        s = self.status()
        logger.info(
            "stats: fill %.2f s, analysis backlog %d, underruns %d, overruns %d, drift corrections %d, "
            "xruns in/out %d/%d, capture errors %d",
            s["fill_s"], s["analysis_backlog"], s["underruns"], s["overruns"], s["drift_corrections"],
            s["xruns_in"], s["xruns_out"], s["capture_errors"],
        )

    def shutdown(self) -> None:
        self.stop_event.set()
        for t in self._threads:
            if t.name != "control":  # blocked on stdin; it's a daemon thread
                t.join(timeout=2.0)
        for pcm in (self._capture, self._playback):
            if pcm:
                pcm.close()
        logger.info("Stopped")

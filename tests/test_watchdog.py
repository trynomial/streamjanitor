"""Daemon start-up order and capture error handling, with fake ALSA devices."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from streamjanitor import watchdog
from streamjanitor.audio_io import SND_PCM_STREAM_PLAYBACK, AlsaError


class FakeDevices:
    """Stands in for AlsaPcm. Paced like real devices; capture fails `fail_reads` times,
    like a loopback read before the DAC runs. Records what happened in `events`."""

    def __init__(self, fail_reads: int) -> None:
        self.fail_reads = fail_reads
        self.events: list[str] = []
        self.playback = None

    def __call__(self, device, stream, sample_rate, channels, latency_ms=100):
        return FakePcm(self, stream == SND_PCM_STREAM_PLAYBACK, sample_rate, channels)


class FakePcm:
    def __init__(self, devices: FakeDevices, playback: bool, rate: int, channels: int) -> None:
        self.devices, self.is_playback, self.rate, self.channels = devices, playback, rate, channels
        self.xruns = self.frames_written = 0
        if playback:
            devices.playback = self
            devices.events.append("open playback")
        else:
            devices.events.append(f"open capture after {devices.playback.frames_written} frames played")

    def read(self, n):
        time.sleep(n / self.rate)
        if self.devices.fail_reads:
            self.devices.fail_reads -= 1
            raise AlsaError("fake: read failed: Input/output error")
        return np.zeros((n, self.channels))

    def write(self, frames):
        if self.frames_written >= self.rate // 10:  # device buffer full: now paced
            time.sleep(len(frames) / self.rate)
        self.frames_written += len(frames)

    def close(self):
        pass


class TestDaemon(unittest.TestCase):
    def test_playback_runs_before_capture_and_capture_errors_are_survived(self):
        devices = FakeDevices(fail_reads=2)
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / "config.toml"
            config.write_text("[audio]\nsample_rate = 48000\ndelay_s = 0.5\n")
            with mock.patch.object(watchdog, "AlsaPcm", devices), \
                 mock.patch.object(watchdog, "CAPTURE_RETRY_S", (0.05, 0.1)):
                daemon = watchdog.Daemon(config)
                daemon.start()
                time.sleep(1.5)
                status = daemon.status()
                daemon.shutdown()

        self.assertEqual(devices.events[0], "open playback")
        played = int(devices.events[1].split()[3])
        self.assertGreaterEqual(played, 48000 // 10, "capture opened before playback ran")
        self.assertEqual(sum(e.startswith("open capture") for e in devices.events), 3)
        self.assertEqual(status["capture_errors"], 2)
        self.assertIsNone(status["last_error"], "the error must clear once capture works again")
        self.assertEqual(daemon._crashes, [], "a capture error must not kill the daemon")
        self.assertGreater(status["fill_s"], 0.3, "capture must resume after reopening")


if __name__ == "__main__":
    unittest.main()

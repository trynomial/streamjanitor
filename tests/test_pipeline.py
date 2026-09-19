"""End-to-end: the whole live chain on synthetic audio, checking the actual output samples."""

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamjanitor.audiofile import ArrayClip
from streamjanitor.config import AudioConfig, load_config
from streamjanitor.detector import HeadTemplate
from streamjanitor.features import HOP_S, extract_features
from streamjanitor.paths import example_config
from streamjanitor.pipeline import Pipeline

from .synth import degrade, melody

RATE = 48000
PERIOD = 1024


def template_for(target, head_s=3.0):
    frames, _ = extract_features(target.mean(axis=1), RATE)
    width = round(head_s / HOP_S)
    return HeadTemplate("ad", "Ad", frames[:width], 0, len(target) / RATE, tau=0.6)


def run(pipe, stream):
    """Feed period by period like the daemon; returns output aligned with the input."""
    out = np.full_like(stream, np.nan)
    padded = np.concatenate((stream, np.zeros((int(pipe.ring.delay) + 2 * PERIOD, 2), np.float32)))
    for i in range(0, len(padded) - PERIOD + 1, PERIOD):
        pipe.capture(padded[i : i + PERIOD])
        pipe.analyze(padded[i : i + PERIOD])
        frames, pos = pipe.render(PERIOD)
        if pos is not None and pos < len(stream):
            k = min(PERIOD, len(stream) - pos)
            out[pos : pos + k] = frames[:k]
    return out


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        cls.target = melody(8.0, seed=1)
        before, after = melody(12.3, seed=2), melody(10.0, seed=3)
        cls.start = len(before)
        cls.end = cls.start + len(cls.target)
        cls.stream = degrade(np.concatenate((before, cls.target, after)))
        cls.clip = ArrayClip(np.full((4 * RATE, 2), 0.1))

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def check_cut(self, out, cut_start, margin_s=0.05):
        m = int(margin_s * RATE)
        np.testing.assert_allclose(out[: cut_start - m], self.stream[: cut_start - m], atol=1e-6)
        np.testing.assert_allclose(out[cut_start + m : cut_start + 4 * RATE - m], 0.1, atol=1e-4)
        np.testing.assert_allclose(out[cut_start + 4 * RATE + m : self.end - m], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[self.end + m :], self.stream[self.end + m :], atol=1e-6)

    def test_whole_target_replaced_when_delay_covers_detection(self):
        pipe = Pipeline([template_for(self.target)], {"ad": self.clip}, AudioConfig(sample_rate=RATE, delay_s=4.0))
        out = run(pipe, self.stream)
        self.assertFalse(np.isnan(out).any())
        self.check_cut(out, self.start)

    def test_short_delay_leaks_but_ends_on_time(self):
        pipe = Pipeline([template_for(self.target)], {"ad": self.clip}, AudioConfig(sample_rate=RATE, delay_s=2.0))
        out = run(pipe, self.stream)
        m = int(0.05 * RATE)
        np.testing.assert_allclose(out[self.end + m :], self.stream[self.end + m :], atol=1e-6)
        leaked = np.flatnonzero(np.abs(out[self.start : self.end, 0] - self.stream[self.start : self.end, 0]) > 1e-6)
        leak_s = leaked[0] / RATE
        self.assertTrue(1.0 < leak_s < 1.8, f"expected ~1.3 s leak, got {leak_s:.2f}")

    def test_report_only_lets_target_through_but_records_it(self):
        pipe = Pipeline([template_for(self.target)], {"ad": self.clip}, AudioConfig(sample_rate=RATE, delay_s=4.0))
        pipe.set_mode("report")
        out = run(pipe, self.stream)
        np.testing.assert_allclose(out, self.stream, atol=1e-6)
        self.assertEqual([(h["id"], h["mode"]) for h in pipe.history], [("ad", "report")])

    def test_passthrough_does_not_analyse(self):
        pipe = Pipeline([template_for(self.target)], {"ad": self.clip}, AudioConfig(sample_rate=RATE, delay_s=4.0))
        pipe.set_mode("passthrough")
        pipe.detector.process = None  # any analysis would crash
        out = run(pipe, self.stream)
        np.testing.assert_allclose(out, self.stream, atol=1e-6)
        self.assertEqual(list(pipe.history), [])

    def test_recognition_resumes_after_passthrough(self):
        # Pass-through for the first 5 s, then replace: the target (at self.start) must be cut
        # at the right place, i.e. frame indices stay aligned with the samples skipped meanwhile.
        pipe = Pipeline([template_for(self.target)], {"ad": self.clip}, AudioConfig(sample_rate=RATE, delay_s=4.0))
        self.assertGreater(self.start, 7 * RATE)
        pipe.set_mode("passthrough")
        out = np.full_like(self.stream, np.nan)
        for i in range(0, len(self.stream) + pipe.ring.delay + 2 * PERIOD, PERIOD):
            if i >= 5 * RATE and pipe.mode == "passthrough":
                pipe.set_mode("replace")
            chunk = np.zeros((PERIOD, 2), np.float32)
            part = self.stream[i : i + PERIOD]
            chunk[: len(part)] = part
            pipe.capture(chunk)
            pipe.analyze(chunk)
            frames, pos = pipe.render(PERIOD)
            if pos is not None and pos < len(self.stream):
                k = min(PERIOD, len(self.stream) - pos)
                out[pos : pos + k] = frames[:k]
        self.check_cut(out, self.start)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            Pipeline([], {}, AudioConfig(sample_rate=RATE)).set_mode("bypass")

    def test_report_only_mid_cut_fades_back_to_live(self):
        pipe = Pipeline([template_for(self.target)], {"ad": self.clip}, AudioConfig(sample_rate=RATE, delay_s=4.0))
        out = np.full_like(self.stream, np.nan)
        switch_at = self.start + 2 * RATE  # output position where report only is switched on
        for i in range(0, len(self.stream) + pipe.ring.delay + 2 * PERIOD, PERIOD):
            chunk = np.zeros((PERIOD, 2), np.float32)
            part = self.stream[i : i + PERIOD]
            chunk[: len(part)] = part
            pipe.capture(chunk)
            pipe.analyze(chunk)
            frames, pos = pipe.render(PERIOD)
            if pos is not None and pos < len(self.stream):
                if pipe.mode == "replace" and pos >= switch_at:
                    pipe.set_mode("report")
                k = min(PERIOD, len(self.stream) - pos)
                out[pos : pos + k] = frames[:k]
        m = int(0.05 * RATE)
        np.testing.assert_allclose(out[self.start + m : switch_at], 0.1, atol=1e-4)  # was cutting
        after = switch_at + PERIOD + m
        np.testing.assert_allclose(out[after:], self.stream[after:], atol=1e-6)  # live again
        self.assertLess(np.abs(np.diff(out[switch_at : after, 0])).max(), 0.05, "no click")

    def test_no_cut_without_target(self):
        pipe = Pipeline([template_for(melody(8.0, seed=9))], {}, AudioConfig(sample_rate=RATE, delay_s=4.0))
        out = run(pipe, self.stream)
        np.testing.assert_allclose(out, self.stream, atol=1e-6)


class TestConfig(unittest.TestCase):
    def test_example_config_parses(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.toml"
            path.write_text(example_config())
            cfg = load_config(path)
        self.assertEqual(cfg.audio.sample_rate, 96000)
        self.assertEqual(cfg.library, Path(d) / "library")
        self.assertEqual(cfg.targets, [])

    def test_unknown_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.toml"
            path.write_text('[[targets]]\nfile = "a.npz"\ntau1 = 0.6\n')
            with self.assertRaisesRegex(ValueError, "tau1"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

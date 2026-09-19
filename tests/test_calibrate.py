import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamjanitor.audiofile import WavWriter
from streamjanitor.calibrate import calibrate, recording_features
from streamjanitor.detector import score_trace
from streamjanitor.features import extract_features
from streamjanitor.precompute import make_template

from .synth import degrade, melody

RATE = 48000


class TestCalibrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target = melody(8.0, seed=1)
        cls.template, _ = make_template(target.mean(axis=1), RATE, "t", "T")
        parts = [melody(40.0, seed=2), target, melody(30.0, seed=3), target, melody(20.0, seed=4)]
        cls.episode = degrade(np.concatenate(parts))
        cls.frames, cls.rms = extract_features(cls.episode.mean(axis=1), RATE)

    def test_finds_every_airing_and_suggests_tau_between(self):
        result = calibrate(self.template, self.frames, self.rms)
        self.assertEqual(result.occurrences, 2)
        starts = sorted(m.start_s for m in result.matches if m.is_target)
        np.testing.assert_allclose(starts, [40.0, 78.0], atol=0.03)
        self.assertLess(result.background, 0.6)
        self.assertTrue(result.background < result.suggested_tau < result.weakest)
        self.assertFalse(any("margin" in n for n in result.notes))

    def test_no_suggestion_without_enough_background(self):
        frames, rms = extract_features(degrade(melody(8.0, seed=1)).mean(axis=1), RATE)
        result = calibrate(self.template, frames, rms)
        self.assertIsNone(result.suggested_tau)
        self.assertTrue(any("too little" in n for n in result.notes))

    def test_silent_recording_is_an_error(self):
        frames, rms = extract_features(np.zeros(RATE * 10, np.float32), RATE)
        with self.assertRaises(ValueError):
            calibrate(self.template, frames, rms)

    def test_blocked_score_trace_matches_single_block(self):
        head = self.template.head
        full = score_trace(head, self.frames, self.rms, 0.003, 0.5)
        # force several small blocks by scoring a slice-by-slice reconstruction
        n, w = len(self.frames), len(head)
        parts = [score_trace(head, self.frames[s : s + 700 + w - 1], self.rms[s : s + 700 + w - 1], 0.003, 0.5)[w - 1 :]
                 for s in range(0, n - w + 1, 700)]
        np.testing.assert_allclose(np.concatenate(parts), full[w - 1 :], atol=1e-5)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_streamed_features_match_in_memory(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ep.wav"
            with WavWriter(path, RATE, 2) as w:
                w.write(self.episode)
            frames, _ = recording_features(path, RATE)
        n = min(len(frames), len(self.frames))
        self.assertLessEqual(abs(len(frames) - len(self.frames)), 1)
        np.testing.assert_allclose(frames[:n], self.frames[:n], atol=0.05)  # 16-bit WAV quantization


if __name__ == "__main__":
    unittest.main()

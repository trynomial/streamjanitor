import unittest

import numpy as np

from streamjanitor.audio_io import AudioRingBuffer
from streamjanitor.audiofile import ArrayClip
from streamjanitor.splice import SpliceEngine

LIVE = 0.5  # the live stream is a constant, so live and clip are easy to tell apart


def engine():
    """Splice engine on a 100 ms delay line (rate 1000) with 10-sample crossfades."""
    return SpliceEngine(AudioRingBuffer(2, 1000, delay_s=0.1), 2, 10)


def render_all(eng, total, level=LIVE, period=64):
    """Feed a constant `level` and render period by period, like the daemon; returns the
    output from its first sample."""
    out = []
    while sum(len(o) for o in out) < total:
        eng.ring.write(np.full((period, 2), level))
        frames, pos = eng.render(period)
        if pos is not None:
            out.append(frames)
    return np.concatenate(out)[:total]


class TestSpliceEngine(unittest.TestCase):
    def test_cut_is_sample_accurate_with_crossfades(self):
        eng = engine()
        clip = ArrayClip(np.full((1000, 2), -0.5))
        self.assertEqual(eng.schedule(500, 800, clip), 0)
        out = render_all(eng, 1200)[:, 0]
        np.testing.assert_allclose(out[:500], 0.5)
        np.testing.assert_allclose(out[510:790], -0.5)
        np.testing.assert_allclose(out[800:], 0.5)
        # equal-power crossfade: smooth, no jump larger than one fade step
        self.assertLess(np.max(np.abs(np.diff(out))), 0.2)

    def test_silence_when_no_clip_or_clip_too_short(self):
        eng = engine()
        eng.schedule(100, 400, ArrayClip(np.full((100, 2), 0.5)))
        out = render_all(eng, 500)[:, 0]
        np.testing.assert_allclose(out[110:200], 0.5)  # clip
        np.testing.assert_allclose(out[200:390], 0.0)  # clip over, target still running

    def test_late_detection_starts_now_and_reports_leak(self):
        eng = engine()
        render_all(eng, 300)
        now = eng.ring.read_position
        leak = eng.schedule(200, 600, None)
        self.assertEqual(leak, now - 200)
        out = render_all(eng, 400)[:, 0]
        self.assertAlmostEqual(out[0], 0.5)  # fades in from the current position
        np.testing.assert_allclose(out[10 : 590 - now], 0.0)
        np.testing.assert_allclose(out[600 - now :], 0.5)

    def test_overlapping_cuts_merge(self):
        eng = engine()
        eng.schedule(100, 300, None)
        eng.schedule(250, 500, None)
        self.assertEqual(len(eng.pending), 1)
        self.assertEqual((eng.pending[0].start, eng.pending[0].end), (100, 500))

    def test_output_is_clipped(self):
        eng = engine()
        eng.schedule(0, 1000, ArrayClip(np.full((1000, 2), 1.5)))
        out = render_all(eng, 1000, level=0.9)
        self.assertLessEqual(out.max(), 1.0)


if __name__ == "__main__":
    unittest.main()

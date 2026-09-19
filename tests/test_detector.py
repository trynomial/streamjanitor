import tempfile
import unittest
from pathlib import Path

import numpy as np

from streamjanitor.detector import HeadDetector, HeadTemplate, score_trace
from streamjanitor.features import HOP_S, extract_features

from .synth import degrade, melody

RATE = 48000


def features(x):
    return extract_features(x.mean(axis=1), RATE)


def make_template(target, head_s=3.0, offset=0, tau=0.6):
    frames, _ = features(target)
    width = round(head_s / HOP_S)
    return HeadTemplate("t", "Target", frames[offset : offset + width], offset, len(target) / RATE, tau=tau)


def run_live(detector, frames, rms, chunk=7):
    detections = []
    for i in range(0, len(frames), chunk):
        detections += detector.process(frames[i : i + chunk], rms[i : i + chunk])
    return detections


class TestHeadDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = melody(8.0, seed=1)
        cls.before, cls.after = melody(10.0, seed=2), melody(10.0, seed=3)
        cls.stream = degrade(np.concatenate((cls.before, cls.target, cls.after)))
        cls.frames, cls.rms = features(cls.stream)
        cls.true_start = round(10.0 / HOP_S)

    def test_finds_exact_start(self):
        for offset in (0, 60):
            with self.subTest(head_offset=offset):
                t = make_template(self.target, offset=offset)
                dets = run_live(HeadDetector([t]), self.frames, self.rms)
                self.assertEqual(len(dets), 1)
                self.assertLessEqual(abs(dets[0].start_frame - self.true_start), 1)
                self.assertGreater(dets[0].score, 0.8)

    def test_no_detection_on_other_music(self):
        t = make_template(self.target)
        frames, rms = features(degrade(melody(60.0, seed=7)))
        self.assertEqual(run_live(HeadDetector([t]), frames, rms), [])
        self.assertLess(score_trace(t.head, frames, rms, 0.003, 0.5).max(), 0.5)

    def test_live_scores_match_offline_trace(self):
        t = make_template(self.target, tau=2.0)  # never fires, so the target is never muted
        trace = score_trace(t.head, self.frames, self.rms, 0.003, 0.5)
        det = HeadDetector([t])
        live = []
        for f, r in zip(self.frames, self.rms, strict=True):
            det.process(f[None], r[None])
            live.append(det.last_scores.get("t", 0.0))
        np.testing.assert_allclose(live, trace, atol=1e-4)

    def test_muted_while_target_plays_then_rearmed(self):
        t = make_template(self.target)
        stream = degrade(np.concatenate((self.before, self.target, self.target, self.after)))
        frames, rms = features(stream)
        dets = run_live(HeadDetector([t]), frames, rms)
        self.assertEqual([d.start_frame for d in dets], [self.true_start, self.true_start + 400])

    def test_silence_never_matches(self):
        t = make_template(self.target)
        quiet = np.concatenate((self.before, 1e-5 * self.target))
        frames, rms = features(quiet)
        self.assertEqual(run_live(HeadDetector([t]), frames, rms), [])

    def test_many_targets_of_different_head_lengths(self):
        a, b = melody(8.0, seed=11), melody(6.0, seed=12)
        stream = degrade(np.concatenate((self.before, a, self.after, b, self.before)))
        frames, rms = features(stream)
        heads = [
            make_template(a, head_s=3.0),
            HeadTemplate("b", "B", features(b)[0][:100], 0, 6.0),  # 2 s head: another width
            HeadTemplate("absent", "Absent", features(melody(5.0, seed=13))[0][:150], 0, 5.0),
        ]
        heads[0].target_id = "a"
        dets = run_live(HeadDetector(heads), frames, rms)
        self.assertEqual(sorted((d.target_id, d.start_frame) for d in dets),
                         [("a", 500), ("b", 500 + 400 + 500)])

    def test_set_targets_mid_stream_keeps_alignment(self):
        t = make_template(self.target)
        other = HeadTemplate("o", "Other", features(melody(4.0, seed=8))[0][:100], 0, 4.0)
        det = HeadDetector([other])
        split = self.true_start + 50  # swap while the head of the target is going by
        dets = run_live(det, self.frames[:split], self.rms[:split])
        det.set_targets([t, other])
        dets += run_live(det, self.frames[split:], self.rms[split:])
        self.assertEqual([(d.target_id, d.start_frame) for d in dets], [("t", self.true_start)])

    def test_template_roundtrip(self):
        t = make_template(self.target, offset=12, tau=0.7)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.npz"
            t.save(path)
            loaded = HeadTemplate.load(path, name="Renamed", tau=None)
        np.testing.assert_allclose(loaded.head, t.head, atol=1e-6)
        self.assertEqual((loaded.head_offset, loaded.name, loaded.tau), (12, "Renamed", 0.6))
        self.assertAlmostEqual(loaded.duration_s, 8.0)


if __name__ == "__main__":
    unittest.main()

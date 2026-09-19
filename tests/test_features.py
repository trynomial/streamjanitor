import unittest

import numpy as np

from streamjanitor.features import (
    Decimator,
    LogMelExtractor,
    create_mel_filterbank,
    extract_features,
)


def tone(freq, seconds, rate):
    return np.sin(2 * np.pi * freq * np.arange(int(seconds * rate)) / rate).astype(np.float32)


class TestDecimator(unittest.TestCase):
    def test_rejects_aliases_and_keeps_passband(self):
        def gain(freq, rate):  # output/input RMS of a sine through the decimator
            out = Decimator(rate).process(tone(freq, 1.0, rate))[200:]  # skip the filter's warm-up
            return float(np.sqrt(np.mean(out**2))) / np.sqrt(0.5)

        for rate in (48000, 96000):
            with self.subTest(rate=rate):
                self.assertAlmostEqual(gain(1000, rate), 1.0, delta=0.01)
                for freq in (8500, 10000, 13000, 20000, 45000)[: 4 if rate == 48000 else 5]:
                    self.assertLess(gain(freq, rate), 1e-3, f"{freq} Hz leaks through")

    def test_streaming_equals_offline(self):
        x = np.random.default_rng(0).normal(size=48000).astype(np.float32)
        whole = Decimator(48000).process(x)
        dec = Decimator(48000)
        chunks = np.concatenate([dec.process(c) for c in np.array_split(x, 37)])
        np.testing.assert_allclose(chunks, whole, atol=1e-6)

    def test_rejects_unsupported_rate(self):
        with self.assertRaises(ValueError):
            Decimator(44100)


class TestLogMel(unittest.TestCase):
    def test_streaming_equals_offline(self):
        x = np.random.default_rng(1).normal(size=16000 * 3).astype(np.float32)
        frames, rms = LogMelExtractor().process(x)
        ext = LogMelExtractor()
        parts = [ext.process(c) for c in np.array_split(x, 23)]
        np.testing.assert_allclose(np.concatenate([p[0] for p in parts]), frames, atol=1e-4)
        np.testing.assert_allclose(np.concatenate([p[1] for p in parts]), rms, atol=1e-6)
        self.assertEqual(len(frames), (len(x) - 1600) // 320 + 1)

    def test_level_invariance(self):
        t = np.arange(48000) / 48000
        x = sum(0.5**k * np.sin(2 * np.pi * 220 * (k + 1) * t) for k in range(6)).astype(np.float32)
        a, _ = extract_features(x, 48000)
        b, _ = extract_features(5 * x, 48000)
        np.testing.assert_allclose(a, b, atol=1e-3)

    def test_filterbank(self):
        fb = create_mel_filterbank()
        self.assertEqual(fb.shape, (60, 1025))
        self.assertTrue(np.all(fb >= 0))
        self.assertTrue(np.all(fb.sum(axis=1) > 0), "every mel band must cover at least one bin")


if __name__ == "__main__":
    unittest.main()

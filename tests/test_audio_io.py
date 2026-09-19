import unittest

import numpy as np

from streamjanitor.audio_io import AudioRingBuffer, float_to_int32, int32_to_float
from streamjanitor.config import AudioConfig
from streamjanitor.pipeline import Pipeline


def ramp(start, n):
    """Frames whose value is their absolute index: makes gaps/reorders visible."""
    v = np.arange(start, start + n, dtype=np.float32)
    return np.repeat(v[:, None], 2, axis=1)


class TestAudioRingBuffer(unittest.TestCase):
    def test_primes_then_plays_in_order(self):
        rb = AudioRingBuffer(2, 1000, delay_s=0.2)
        rb.write(ramp(0, 150))
        out, pos = rb.read(50)
        self.assertIsNone(pos)
        self.assertTrue(np.all(out == 0))
        rb.write(ramp(150, 100))
        out, pos = rb.read(50)
        self.assertEqual(pos, 0)
        np.testing.assert_array_equal(out[:, 0], np.arange(50))

    def test_jitter_does_not_create_gaps(self):
        rate, period = 48000, 1024
        rb = AudioRingBuffer(2, rate, delay_s=1.0)
        rng = np.random.default_rng(0)
        T = period / rate
        events = [(k * T + rng.normal(0, 0.004), "w") for k in range(2000)]
        events += [(k * T + T / 2 + rng.normal(0, 0.004), "r") for k in range(2000)]
        written, played = 0, []
        for _, kind in sorted(events):
            if kind == "w":
                rb.write(ramp(written, period))
                written += period
            else:
                out, pos = rb.read(period)
                if pos is not None:
                    played.append(out[:, 0])
        played = np.concatenate(played)
        np.testing.assert_array_equal(played, np.arange(len(played)))
        self.assertEqual(rb.underruns, 0)

    def test_underrun_reprimes(self):
        rb = AudioRingBuffer(2, 1000, delay_s=0.1)
        rb.write(ramp(0, 120))
        out, _ = rb.read(200)
        self.assertEqual(rb.underruns, 1)
        np.testing.assert_array_equal(out[:120, 0], np.arange(120))
        self.assertTrue(np.all(out[120:] == 0))
        rb.write(ramp(120, 50))
        self.assertIsNone(rb.read(10)[1], "must wait for the full delay again")

    def test_overrun_drops_oldest(self):
        rb = AudioRingBuffer(2, 1000, delay_s=0.1)
        rb.write(ramp(0, rb.capacity + 300))
        self.assertEqual(rb.overruns, 1)
        out, pos = rb.read(10)
        self.assertEqual(pos, 300)
        self.assertEqual(out[0, 0], 300)

    def test_drift_correction_both_directions(self):
        rate, period = 1000, 10
        for extra, direction in ((1, "capture faster"), (-1, "playback faster")):
            with self.subTest(direction):
                rb = AudioRingBuffer(2, rate, delay_s=0.5, drift_tolerance_s=0.05)
                written = 0
                for k in range(3000):
                    n = period + (extra if k % 5 == 0 else 0)  # 2% clock mismatch
                    rb.write(ramp(written, n))
                    written += n
                    rb.read(period)
                self.assertGreater(rb.drift_corrections, 0)
                self.assertEqual(rb.underruns, 0)
                self.assertLessEqual(abs(rb.fill - rb.delay), rb.tolerance + 2 * period)


class TestBitPerfect(unittest.TestCase):
    """Outside cuts the daemon must hand the DAC exactly the samples it captured."""

    def samples(self, bits, n=96000 * 3):
        rng = np.random.default_rng(bits)
        k = rng.integers(-(2 ** (bits - 1)), 2 ** (bits - 1), size=(n, 2), dtype=np.int64)
        k[:10] = [[-(2 ** (bits - 1)), 2 ** (bits - 1) - 1]] * 10  # full scale, both signs
        return (k << (32 - bits)).astype(np.int32)  # left-aligned in S32, as ALSA delivers it

    def test_conversion_roundtrip_is_exact(self):
        for bits in (16, 24, 32):
            with self.subTest(bits=bits):
                x = self.samples(bits)
                np.testing.assert_array_equal(float_to_int32(int32_to_float(x)), x)

    def test_out_of_range_is_clipped(self):
        np.testing.assert_array_equal(float_to_int32(np.array([1.5, -1.5, 1.0])),
                                      [2**31 - 1, -(2**31), 2**31 - 1])

    def test_pipeline_passthrough_is_bit_perfect(self):
        period, x = 1024, self.samples(32)  # full 32-bit samples, as Mopidy writes into the loopback
        pipe = Pipeline([], {}, AudioConfig(sample_rate=96000, delay_s=0.5, period_size=period))
        out = np.zeros_like(x)
        padded = np.concatenate((x, np.zeros((pipe.ring.delay + 2 * period, 2), np.int32)))
        for i in range(0, len(padded) - period + 1, period):
            chunk = int32_to_float(padded[i : i + period])
            pipe.capture(chunk)
            pipe.analyze(chunk)
            frames, pos = pipe.render(period)
            if pos is not None and pos < len(x):
                k = min(period, len(x) - pos)
                out[pos : pos + k] = float_to_int32(frames[:k])
        np.testing.assert_array_equal(out, x)


if __name__ == "__main__":
    unittest.main()

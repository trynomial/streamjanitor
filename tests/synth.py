"""Synthetic test audio: random melodies that are spectrally distinct per seed."""

import numpy as np


def melody(seconds: float, seed: int, sample_rate: int = 48000) -> np.ndarray:
    """Stereo float32 (n, 2): random notes (150-400 ms, 4 harmonics, decaying envelope)."""
    rng = np.random.default_rng(seed)
    n = round(seconds * sample_rate)
    out = np.zeros(n, dtype=np.float64)
    pos = 0
    while pos < n:
        length = int(rng.uniform(0.15, 0.4) * sample_rate)
        f0 = 440.0 * 2 ** ((rng.integers(45, 85) - 69) / 12)
        t = np.arange(min(length, n - pos)) / sample_rate
        note = sum(0.6**h * np.sin(2 * np.pi * f0 * (h + 1) * t) for h in range(4))
        out[pos : pos + len(t)] += note * np.exp(-3.0 * t) * rng.uniform(0.5, 1.0)
        pos += length
    out = 0.25 * out / np.max(np.abs(out))
    return np.repeat(out[:, None], 2, axis=1).astype(np.float32)


def degrade(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Imitate a different encoding of the same audio: gain, dull top end, noise floor."""
    rng = np.random.default_rng(seed)
    y = 0.7 * x
    y = 0.5 * (y + np.roll(y, 1, axis=0))
    y = y + rng.normal(0, 0.002, y.shape)
    return y.astype(np.float32)

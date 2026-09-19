"""Feature extraction shared by the PC tools and the Pi daemon.

Templates and live audio must go through exactly this code path, decimator
included: features computed with a different resampler do not match.

Pipeline: mono at the pipeline rate (48 or 96 kHz) -> Decimator -> 16 kHz ->
LogMelExtractor -> one 60-bin log-mel frame every 20 ms, z-normalized per frame
(invariant to level and to most of the colouring added by lossy codecs).

Frame k covers 16 kHz samples [k*HOP, k*HOP + WINDOW), i.e. pipeline samples
starting at k*HOP*factor, counted from the first sample fed to the Decimator.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FEATURE_RATE = 16000
HOP = 320  # 20 ms
WINDOW = 1600  # 100 ms
FFT_SIZE = 2048
NUM_MELS = 60
MEL_FMIN = 80.0
MEL_FMAX = 7800.0
HOP_S = HOP / FEATURE_RATE


def decimation_factor(sample_rate: int) -> int:
    if sample_rate % FEATURE_RATE:
        raise ValueError(
            f"sample rate {sample_rate} is not a multiple of {FEATURE_RATE} Hz (use 48000 or 96000)"
        )
    return sample_rate // FEATURE_RATE


PASS_HZ, STOP_HZ, STOP_DB = 7000.0, 8200.0, 60.0  # what reaches the 16 kHz features


def design_lowpass(sample_rate: int, stop_hz: float = STOP_HZ) -> np.ndarray:
    """Kaiser-windowed FIR: flat up to PASS_HZ, >= STOP_DB rejection from stop_hz.

    The defaults suit the last step to 16 kHz: aliases fold below 8 kHz only after heavy
    attenuation, so the mel bands (up to 7.8 kHz) stay clean.
    """
    beta = 0.1102 * (STOP_DB - 8.7)
    delta_w = 2.0 * np.pi * (stop_hz - PASS_HZ) / sample_rate
    num_taps = int(np.ceil((STOP_DB - 8.0) / (2.285 * delta_w))) | 1  # odd -> symmetric
    cutoff = (PASS_HZ + stop_hz) / 2.0 / sample_rate
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * n) * np.kaiser(num_taps, beta)
    return (taps / taps.sum()).astype(np.float32)


class _FirDecimator:
    """Streaming FIR low-pass + decimation by `factor`; only the kept outputs are computed."""

    def __init__(self, taps: np.ndarray, factor: int) -> None:
        self.taps, self.factor = taps, factor
        self._pending = np.zeros(0, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate((self._pending, np.asarray(x, dtype=np.float32)))
        n_taps = len(self.taps)
        if len(buf) < n_taps:
            self._pending = buf
            return np.zeros(0, dtype=np.float32)
        n_out = (len(buf) - n_taps) // self.factor + 1
        # np.convolve computes every output and we keep 1 in `factor`: still ~2x faster than
        # a strided matrix product on a Pi. Taps are symmetric, so convolution == correlation.
        y = np.convolve(buf, self.taps, "valid")[: n_out * self.factor : self.factor]
        self._pending = buf[n_out * self.factor :]
        return y.astype(np.float32, copy=False)


class Decimator:
    """Streaming decimator to 16 kHz.

    From 96 kHz it works in two steps: a short filter halves the rate (it only has to reject
    what would fold below STOP_HZ, i.e. everything above 48 kHz - STOP_HZ), then the full
    filter takes 48 kHz to 16 kHz. Same response as one long filter, about half the work.
    """

    def __init__(self, sample_rate: int) -> None:
        factor = decimation_factor(sample_rate)
        self._stages: list[tuple[_FirDecimator, int]] = []
        rate = sample_rate
        if factor % 2 == 0 and factor > 2:
            half = rate // 2
            self._stages.append((_FirDecimator(design_lowpass(rate, stop_hz=half - STOP_HZ), 2), rate))
            rate, factor = half, factor // 2
        self._stages.append((_FirDecimator(design_lowpass(rate), factor), rate))

    def stages(self) -> list[tuple[np.ndarray, int]]:
        """(taps, input rate) of each step."""
        return [(stage.taps, rate) for stage, rate in self._stages]

    def process(self, x: np.ndarray) -> np.ndarray:
        for stage, _ in self._stages:
            x = stage.process(x)
        return x


def create_mel_filterbank() -> np.ndarray:
    """Triangular mel filterbank, shape (NUM_MELS, FFT_SIZE // 2 + 1), area-normalized."""

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    num_bins = FFT_SIZE // 2 + 1
    hz_points = mel_to_hz(np.linspace(hz_to_mel(MEL_FMIN), hz_to_mel(MEL_FMAX), NUM_MELS + 2))
    bin_freqs = np.arange(num_bins) * FEATURE_RATE / FFT_SIZE

    weights = np.zeros((NUM_MELS, num_bins), dtype=np.float32)
    for m in range(NUM_MELS):
        left, center, right = hz_points[m : m + 3]
        rising = (bin_freqs - left) / (center - left)
        falling = (right - bin_freqs) / (right - center)
        weights[m] = np.maximum(0.0, np.minimum(rising, falling))
    weights *= (2.0 / (hz_points[2:] - hz_points[:-2]))[:, np.newaxis]
    return weights


class LogMelExtractor:
    """Streaming 16 kHz audio -> z-normalized log-mel frames + per-frame RMS."""

    def __init__(self) -> None:
        self.window = np.hanning(WINDOW).astype(np.float32)
        self.filterbank_t = create_mel_filterbank().T.copy()
        self._residual = np.zeros(0, dtype=np.float32)

    def process(self, x16k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (frames (N, NUM_MELS), rms (N,)) for every complete frame available."""
        buf = np.concatenate((self._residual, np.asarray(x16k, dtype=np.float32)))
        if len(buf) < WINDOW:
            self._residual = buf
            return np.zeros((0, NUM_MELS), np.float32), np.zeros(0, np.float32)

        n = (len(buf) - WINDOW) // HOP + 1
        chunks = sliding_window_view(buf, WINDOW)[: n * HOP : HOP]
        self._residual = buf[n * HOP :]

        rms = np.sqrt(np.mean(chunks * chunks, axis=1))
        spectrum = np.fft.rfft(chunks * self.window, n=FFT_SIZE, axis=1)
        power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
        mel = power @ self.filterbank_t
        # Floor 60 dB below the frame's loudest band: keeps the features level-invariant and
        # ignores near-empty bands, where codecs differ the most.
        floor = np.maximum(mel.max(axis=1, keepdims=True) * 1e-6, 1e-12)
        log_mel = np.log(np.maximum(mel, floor))

        mean = log_mel.mean(axis=1, keepdims=True)
        std = log_mel.std(axis=1, keepdims=True)
        frames = (log_mel - mean) / (std + 1e-6)
        return frames.astype(np.float32), rms.astype(np.float32)


def extract_features(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Offline helper: whole mono signal at the pipeline rate -> (frames, rms)."""
    return LogMelExtractor().process(Decimator(sample_rate).process(mono))

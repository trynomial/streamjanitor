"""Delay line and ALSA PCM access.

AudioRingBuffer is indexed by absolute input sample: sample n is the n-th frame
ever captured. Detections are expressed in the same index, so the splice engine
can place cuts exactly on the delayed output.
"""

import ctypes
import ctypes.util
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class AudioRingBuffer:
    """Thread-safe delay line between capture (writer) and playback (reader).

    Samples are float64, which holds any S32 sample exactly: what goes in comes out
    bit for bit. Playback starts once `delay` frames are buffered and then drains
    whatever is there, so scheduling jitter between threads does not cause gaps.
    If the capture and playback clocks drift apart, one sample per read is skipped
    or repeated to keep the fill near `delay` (never needed when the loopback is
    clocked by the DAC via snd-aloop's timer_source; `drift_corrections` counts them).
    """

    def __init__(
        self,
        channels: int,
        sample_rate: int,
        delay_s: float,
        drift_tolerance_s: float = 0.15,  # well above the fill swings of 25 ms device periods
    ) -> None:
        self.channels = channels
        self.delay = round(delay_s * sample_rate)
        self.tolerance = round(drift_tolerance_s * sample_rate)
        self.capacity = self.delay + 2 * sample_rate
        self._buf = np.zeros((self.capacity, channels), dtype=np.float64)
        self._written = 0  # absolute index of the next captured sample
        self._read = 0  # absolute index of the next sample to play
        self._primed = False
        self._lock = threading.Lock()

        self.underruns = 0
        self.overruns = 0
        self.drift_corrections = 0

    @property
    def read_position(self) -> int:
        """Absolute index of the next input sample that will be played."""
        with self._lock:
            return self._read

    @property
    def fill(self) -> int:
        with self._lock:
            return self._written - self._read

    def write(self, frames: np.ndarray) -> None:
        n = len(frames)
        if n == 0:
            return
        with self._lock:
            if n > self.capacity:
                frames = frames[-self.capacity :]
                self._written += n - self.capacity
                n = self.capacity
            excess = self._written + n - self._read - self.capacity
            if excess > 0:  # playback stalled: forget the oldest audio
                self._read += excess
                self.overruns += 1
            start = self._written % self.capacity
            first = min(n, self.capacity - start)
            self._buf[start : start + first] = frames[:first]
            self._buf[: n - first] = frames[first:]
            self._written += n

    def read(self, n: int) -> tuple[np.ndarray, int | None]:
        """Returns (frames, absolute index of frames[0]); index is None while priming."""
        with self._lock:
            stored = self._written - self._read
            if not self._primed:
                if stored < self.delay:
                    return np.zeros((n, self.channels)), None
                self._primed = True

            take = n
            if stored > self.delay + self.tolerance:
                take = n + 1
            elif stored < self.delay - self.tolerance:
                take = n - 1
            if take != n:
                self.drift_corrections += 1

            pos = self._read
            avail = min(take, stored)
            out = self._copy_out(pos, avail)
            self._read += avail

            if avail < take:
                # Capture stopped: pad with silence and re-prime to restore the full delay.
                self.underruns += 1
                self._primed = False
                return np.vstack((out, np.zeros((n - avail, self.channels))))[:n], pos
            if take == n + 1:
                out = np.delete(out, n // 2, axis=0)
            elif take == n - 1:
                out = np.insert(out, n // 2, out[n // 2 - 1], axis=0)
            return out, pos

    def _copy_out(self, pos: int, n: int) -> np.ndarray:
        start = pos % self.capacity
        first = min(n, self.capacity - start)
        return np.concatenate((self._buf[start : start + first], self._buf[: n - first]))


# --- ALSA PCM through libasound (no compiled Python dependency) ---

SND_PCM_STREAM_PLAYBACK = 0
SND_PCM_STREAM_CAPTURE = 1
SND_PCM_ACCESS_RW_INTERLEAVED = 3
SND_PCM_FORMAT_S32_LE = 10
INT32_FULL_SCALE = 2.0**31  # a power of two: scaling is exact, so audio passes bit-perfect


def int32_to_float(samples: np.ndarray) -> np.ndarray:
    """S32 samples -> float64 in [-1, 1). Exact for any 32-bit sample (float64 has a 53-bit mantissa)."""
    return samples.astype(np.float64) * (1.0 / INT32_FULL_SCALE)


def float_to_int32(frames: np.ndarray) -> np.ndarray:
    """float [-1, 1] -> S32, clipped; the exact inverse of int32_to_float."""
    scaled = np.rint(np.asarray(frames, dtype=np.float64) * INT32_FULL_SCALE)
    return np.ascontiguousarray(np.clip(scaled, -INT32_FULL_SCALE, INT32_FULL_SCALE - 1), dtype=np.int32)


_asound = None


def _libasound() -> ctypes.CDLL:
    global _asound
    if _asound is None:
        lib = ctypes.CDLL(ctypes.util.find_library("asound") or "libasound.so.2")
        pcm_p = ctypes.c_void_p
        lib.snd_pcm_open.argtypes = [ctypes.POINTER(pcm_p), ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.snd_pcm_open.restype = ctypes.c_int
        lib.snd_pcm_set_params.argtypes = [
            pcm_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.c_uint,
        ]
        lib.snd_pcm_set_params.restype = ctypes.c_int
        for fn in (lib.snd_pcm_readi, lib.snd_pcm_writei):
            fn.argtypes = [pcm_p, ctypes.c_void_p, ctypes.c_ulong]
            fn.restype = ctypes.c_long
        sw_p = ctypes.c_void_p
        lib.snd_pcm_sw_params_malloc.argtypes = [ctypes.POINTER(sw_p)]
        lib.snd_pcm_sw_params_free.argtypes = [sw_p]
        lib.snd_pcm_sw_params_current.argtypes = [pcm_p, sw_p]
        lib.snd_pcm_sw_params_set_start_threshold.argtypes = [pcm_p, sw_p, ctypes.c_ulong]
        lib.snd_pcm_sw_params.argtypes = [pcm_p, sw_p]
        for fn in (lib.snd_pcm_sw_params_malloc, lib.snd_pcm_sw_params_current,
                   lib.snd_pcm_sw_params_set_start_threshold, lib.snd_pcm_sw_params):
            fn.restype = ctypes.c_int
        lib.snd_pcm_recover.argtypes = [pcm_p, ctypes.c_int, ctypes.c_int]
        lib.snd_pcm_recover.restype = ctypes.c_int
        lib.snd_pcm_close.argtypes = [pcm_p]
        lib.snd_pcm_close.restype = ctypes.c_int
        lib.snd_strerror.argtypes = [ctypes.c_int]
        lib.snd_strerror.restype = ctypes.c_char_p
        _asound = lib
    return _asound


class AlsaError(RuntimeError):
    pass


class AlsaPcm:
    """Blocking interleaved float64 <-> S32_LE PCM, bit-exact (use plughw: to adapt to the device)."""

    def __init__(
        self,
        device: str,
        stream: int,
        sample_rate: int,
        channels: int,
        latency_ms: int = 100,
    ) -> None:
        self._lib = _libasound()
        self.device = device
        self.channels = channels
        self.xruns = 0
        self.frames_written = 0
        self._pcm = ctypes.c_void_p()
        self._check(self._lib.snd_pcm_open(ctypes.byref(self._pcm), device.encode(), stream, 0), "open")
        self._check(
            self._lib.snd_pcm_set_params(
                self._pcm,
                SND_PCM_FORMAT_S32_LE,
                SND_PCM_ACCESS_RW_INTERLEAVED,
                channels,
                sample_rate,
                1,  # allow ALSA resampling (plughw)
                latency_ms * 1000,
            ),
            "set_params",
        )
        if stream == SND_PCM_STREAM_CAPTURE:
            # set_params makes a stream start only once a whole buffer is requested; reads are
            # smaller, so a capture stream would never start (and the read times out with EIO).
            self._set_start_threshold(1)

    def _set_start_threshold(self, frames: int) -> None:
        sw = ctypes.c_void_p()
        self._check(self._lib.snd_pcm_sw_params_malloc(ctypes.byref(sw)), "sw_params_malloc")
        try:
            self._check(self._lib.snd_pcm_sw_params_current(self._pcm, sw), "sw_params_current")
            self._check(self._lib.snd_pcm_sw_params_set_start_threshold(self._pcm, sw, frames), "set_start_threshold")
            self._check(self._lib.snd_pcm_sw_params(self._pcm, sw), "sw_params")
        finally:
            self._lib.snd_pcm_sw_params_free(sw)

    def _check(self, err: int, what: str) -> None:
        if err < 0:
            raise AlsaError(f"{self.device}: {what} failed: {self._lib.snd_strerror(err).decode()}")

    def _recover(self, err: int, what: str) -> None:
        self.xruns += 1
        logger.warning("%s: %s xrun/error (%s), recovering", self.device, what, self._lib.snd_strerror(err).decode())
        self._check(self._lib.snd_pcm_recover(self._pcm, err, 1), "recover")

    def read(self, n: int) -> np.ndarray:
        buf = np.empty((n, self.channels), dtype=np.int32)
        got = 0
        while got < n:
            ret = self._lib.snd_pcm_readi(self._pcm, buf[got:].ctypes.data, n - got)
            if ret < 0:
                self._recover(int(ret), "read")
                break
            got += ret
        return int32_to_float(buf[:got])

    def write(self, frames: np.ndarray) -> None:
        data = float_to_int32(frames)
        done = 0
        while done < len(data):
            ret = self._lib.snd_pcm_writei(self._pcm, data[done:].ctypes.data, len(data) - done)
            if ret < 0:
                self._recover(int(ret), "write")
                continue
            done += ret
        self.frames_written += done

    def close(self) -> None:
        if self._pcm:
            self._lib.snd_pcm_close(self._pcm)
            self._pcm = ctypes.c_void_p()

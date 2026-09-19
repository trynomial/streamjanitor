"""Audio file decoding/encoding via ffmpeg, and replacement clips read from disk."""

import mmap
import subprocess
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

# High-quality resampling (SoX) for audio that is played; analysis doesn't need it.
HQ_RESAMPLE = ["-af", "aresample=resampler=soxr:precision=28"]


def _ffmpeg_cmd(
    path: Path,
    sample_rate: int,
    channels: int,
    start_s: float = 0.0,
    duration_s: float | None = None,
    hq: bool = False,
) -> list[str]:
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    return cmd + (HQ_RESAMPLE if hq else []) + ["-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels), "-"]


def decode(
    path: str | Path,
    sample_rate: int,
    channels: int,
    start_s: float = 0.0,
    duration_s: float | None = None,
    hq: bool = False,
) -> np.ndarray:
    """File (or the [start_s, start_s + duration_s) part of it) -> float32 (n, channels).

    hq: resample with SoX (falls back to ffmpeg's default resampler if it isn't built in).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cmd = _ffmpeg_cmd(path, sample_rate, channels, start_s, duration_s, hq)
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode and hq:
        return decode(path, sample_rate, channels, start_s, duration_s, hq=False)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed on {path.name}: {proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, channels)


def write_flac(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """float32 (n, channels) -> lossless FLAC (for keeping target sources compact)."""
    channels = audio.shape[1]
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(sample_rate),
           "-ac", str(channels), "-i", "-", "-sample_fmt", "s16", "-c:a", "flac", str(path)]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(audio, np.float32).tobytes(), capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed writing {path}: {proc.stderr.decode(errors='replace').strip()}")


def decode_chunks(path: str | Path, sample_rate: int, channels: int, chunk: int) -> Iterator[np.ndarray]:
    """Stream a file as float32 (<= chunk, channels) blocks without loading it all."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    bytes_per_chunk = chunk * channels * 4
    with subprocess.Popen(_ffmpeg_cmd(path, sample_rate, channels), stdout=subprocess.PIPE) as proc:
        while True:
            data = proc.stdout.read(bytes_per_chunk)
            if not data:
                break
            usable = len(data) - len(data) % (channels * 4)
            yield np.frombuffer(data[:usable], dtype=np.float32).reshape(-1, channels)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed on {path}")


class Clip:
    """Replacement audio at the pipeline rate, read on demand as float64 frames.

    The last `fade` frames are faded out on the fly, so a clip never ends with a click.
    """

    channels: int

    def __len__(self) -> int:
        raise NotImplementedError

    def _raw(self, start: int, n: int) -> np.ndarray:
        raise NotImplementedError

    def prefetch(self) -> None:
        """Hint that playback is about to start (e.g. load it from disk)."""

    def frames(self, start: int, n: int) -> np.ndarray:
        n = max(0, min(n, len(self) - start))
        out = self._raw(start, n)
        tail = len(self) - self.fade
        if n and start + n > tail:
            idx = np.arange(start, start + n)
            gain = np.clip((len(self) - idx) / max(1, self.fade), 0.0, 1.0)
            out = out * gain[:, None]
        return out


class ArrayClip(Clip):
    def __init__(self, audio: np.ndarray, fade: int = 0) -> None:
        self.audio = np.asarray(audio)
        self.channels = self.audio.shape[1]
        self.fade = min(fade, len(self.audio))

    def __len__(self) -> int:
        return len(self.audio)

    def _raw(self, start: int, n: int) -> np.ndarray:
        return self.audio[start : start + n].astype(np.float64)


class WavClip(Clip):
    """Integer PCM WAV (16/24/32 bit) memory-mapped: no RAM needed for long clips."""

    def __init__(self, path: str | Path, fade: int = 0) -> None:
        self.path = Path(path)
        self.channels, self.sample_rate, self.width, offset, size = wav_layout(self.path)
        with self.path.open("rb") as f:
            self._map = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._offset = offset
        self._frames = min(size, len(self._map) - offset) // (self.width * self.channels)
        self.fade = min(fade, self._frames)

    def __len__(self) -> int:
        return self._frames

    def prefetch(self) -> None:
        if hasattr(self._map, "madvise"):  # Linux: start reading it from disk in the background
            self._map.madvise(mmap.MADV_WILLNEED)

    def _raw(self, start: int, n: int) -> np.ndarray:
        b = self.width
        raw = np.frombuffer(self._map, dtype=np.uint8, count=n * b * self.channels,
                            offset=self._offset + start * b * self.channels)
        if b == 2:
            x = raw.view("<i2").astype(np.float64) / 2**15
        elif b == 4:
            x = raw.view("<i4").astype(np.float64) / 2**31
        else:  # 24-bit packed: little-endian 3 bytes, sign-extended through the top byte
            t = raw.reshape(-1, 3).astype(np.int32)
            x = ((t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16)) << 8 >> 8).astype(np.float64) / 2**23
        return x.reshape(n, self.channels)


def wav_layout(path: Path) -> tuple[int, int, int, int, int]:
    """(channels, sample rate, bytes per sample, data offset, data size) of an integer PCM WAV."""
    with path.open("rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"{path.name}: not a WAV file")
        fmt = None
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                raise ValueError(f"{path.name}: no data chunk")
            cid, size = chunk[:4], int.from_bytes(chunk[4:], "little")
            if cid == b"fmt ":
                fmt = f.read(size)
                if size & 1:
                    f.seek(1, 1)
            elif cid == b"data":
                if fmt is None:
                    raise ValueError(f"{path.name}: data before fmt")
                tag = int.from_bytes(fmt[0:2], "little")
                if tag == 0xFFFE and len(fmt) >= 26:  # WAVE_FORMAT_EXTENSIBLE: real tag in the GUID
                    tag = int.from_bytes(fmt[24:26], "little")
                channels = int.from_bytes(fmt[2:4], "little")
                rate = int.from_bytes(fmt[4:8], "little")
                bits = int.from_bytes(fmt[14:16], "little")
                if tag != 1 or bits not in (16, 24, 32):
                    raise ValueError(f"{path.name}: unsupported WAV encoding (tag {tag}, {bits} bit)")
                return channels, rate, bits // 8, f.tell(), size
            else:
                f.seek(size + (size & 1), 1)


def load_clip(path: str | Path, sample_rate: int, channels: int, fade_frames: int) -> Clip:
    """Replacement clip. WAVs already in the pipeline format are read from disk as they are;
    anything else is decoded (and resampled with SoX) into memory."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            ch, rate, _, _, _ = wav_layout(path)
        except ValueError:
            pass
        else:
            if (ch, rate) == (channels, sample_rate):
                return WavClip(path, fade_frames)
    return ArrayClip(decode(path, sample_rate, channels, hq=True), fade_frames)


class WavWriter:
    """Incremental integer PCM WAV writer (16 or 24 bit) for float frames."""

    def __init__(self, path: str | Path, sample_rate: int, channels: int, bits: int = 24) -> None:
        if bits not in (16, 24):
            raise ValueError("bits must be 16 or 24")
        self.bits = bits
        self._wf = wave.open(str(path), "wb")
        self._wf.setnchannels(channels)
        self._wf.setsampwidth(bits // 8)
        self._wf.setframerate(sample_rate)

    def write(self, frames: np.ndarray) -> None:
        full = 2 ** (self.bits - 1)
        pcm = np.clip(np.rint(np.asarray(frames, dtype=np.float64) * full), -full, full - 1).astype("<i4")
        if self.bits == 16:
            data = pcm.astype("<i2").tobytes()
        else:  # keep the low 3 bytes of each little-endian int32
            data = pcm.reshape(-1, 1).view(np.uint8)[:, :3].tobytes()
        self._wf.writeframes(data)

    def close(self) -> None:
        self._wf.close()

    def __enter__(self) -> "WavWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

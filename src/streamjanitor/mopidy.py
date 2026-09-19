"""Mopidy's [audio] output, pointed at the loopback without touching the audio chain.

The user's own GStreamer chain (converters, resampler, caps: sample rate and format)
is kept as it is; only the sink's device becomes the loopback, plus the ALSA buffer
and period sizes, which change timing but not a single sample. With the loopback
clocked by the DAC (snd-aloop timer_source) the period must equal the DAC's, which
streamjanitor opens with alsa_latency_ms (period = a quarter of it).
"""

import re

LOOPBACK_DEVICE = "plughw:Loopback,0"  # plughw: accepts whatever format the chain produces
DEFAULT_CHAIN = "audioconvert ! audioresample ! audio/x-raw,rate={rate},channels=2,format=S24LE ! alsasink"
TIMING_PROPS = ("device", "buffer-time", "latency-time")


def caps_rate(output: str | None) -> int | None:
    """Sample rate forced by the chain's caps, e.g. 96000 for '...audio/x-raw,rate=96000...'."""
    m = re.search(r"audio/x-raw[^!]*?\brate=(?:\(int\))?(\d+)", output or "")
    return int(m[1]) if m else None


def loopback_output(original: str | None, rate: int, alsa_latency_ms: int = 100) -> str:
    """Value for `output =` that plays `original`'s chain at `rate` into the loopback."""
    chain = original.strip() if original and "alsasink" in original else DEFAULT_CHAIN.format(rate=rate)
    elements = [e.strip() for e in chain.split("!")]
    sink = max(i for i, e in enumerate(elements) if e.split()[0] == "alsasink")

    # The loopback carries one fixed rate: make sure the chain forces it.
    caps = [i for i, e in enumerate(elements[:sink]) if e.startswith("audio/x-raw")]
    if caps:
        c = caps[-1]
        if caps_rate(elements[c]) is not None:
            elements[c] = re.sub(r"\brate=(?:\(int\))?\d+", f"rate={rate}", elements[c])
        else:
            elements[c] += f",rate={rate}"
        if not any(e.split()[0] == "audioresample" for e in elements[:c]):
            elements.insert(c, "audioresample")
            sink += 1
    else:
        elements[sink:sink] = ["audioresample", f"audio/x-raw,rate={rate}"]
        sink += 2

    props = [p for p in elements[sink].split()[1:] if p.split("=")[0] not in TIMING_PROPS]
    elements[sink] = " ".join(
        ["alsasink", *props, f"device={LOOPBACK_DEVICE}",
         f"buffer-time={alsa_latency_ms * 1000}", f"latency-time={alsa_latency_ms * 250}"]
    )
    return " ! ".join(elements)

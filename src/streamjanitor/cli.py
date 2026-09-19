"""streamjanitor command line.

  web                  web interface: Studio (edit a library) and/or On Air (run the daemon)
  precompute     (PC)  target audio -> head template (.npz)
  calibrate      (PC)  scan a long recording (e.g. the archived episode) with a template, suggest tau
  simulate       (PC)  run the full live pipeline on a file and write the resulting audio
  run            (Pi)  live daemon (On Air runs it for you)
  record         (Pi)  capture the loopback to WAV
  init           (Pi)  write a config file (and an empty library) in the standard locations
  mopidy-output  (Pi)  Mopidy's [audio] output for the loopback, keeping its chain
  bench                measure the CPU cost of the live pipeline on this machine
"""

import argparse
import json
import logging
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from .audiofile import WavWriter, decode, decode_chunks
from .calibrate import calibrate, recording_features
from .config import DEFAULT_SAMPLE_RATE, MODES, SAMPLE_RATES, load_config
from .detector import DEFAULT_ENERGY_GATE, DEFAULT_MIN_VOICED, HeadTemplate
from .features import HOP_S
from .library import format_time
from .paths import default_config, default_library, default_studio, example_config
from .precompute import DEFAULT_HEAD_S, detection_latency_s, make_template

logger = logging.getLogger("streamjanitor")


def cmd_precompute(args: argparse.Namespace) -> int:
    audio = decode(args.input, args.rate, 1)[:, 0]
    audio = audio[round(args.start * args.rate) :]
    if args.duration > 0:
        audio = audio[: round(args.duration * args.rate)]

    target_id = args.id or Path(args.input).stem
    offset = None if args.head_offset == "auto" else float(args.head_offset)
    try:
        template, warnings = make_template(audio, args.rate, target_id, args.name or target_id,
                                           args.head, offset, args.gate)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    for w in warnings:
        print(f"Warning: {w}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    template.save(args.output)
    latency = detection_latency_s(template) + 0.05
    print(f"Saved {args.output}: id '{target_id}', head {template.width * HOP_S:.2f} s starting at "
          f"+{template.head_offset * HOP_S:.2f} s, duration {template.duration_s:.2f} s")
    print(f"Recognised ~{latency:.1f} s after the target starts: use delay_s >= {latency + 0.3:.1f} for zero leak.")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    template = HeadTemplate.load(args.target)
    print(f"Template '{template.name}': head {template.width * HOP_S:.2f} s at "
          f"+{template.head_offset * HOP_S:.2f} s, duration {template.duration_s:.1f} s")

    frames, rms = recording_features(args.input, args.rate)
    print(f"Recording: {format_time(len(frames) * HOP_S)}")
    try:
        result = calibrate(template, frames, rms, args.gate, args.min_voiced, args.top)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print("\nBest matches (start time of the target in the recording):")
    for rank, m in enumerate(result.matches, 1):
        print(f"  {rank:2d}. {format_time(m.start_s, decimals=2)}  score {m.score:.3f}{'  <- target' if m.is_target else ''}")
    print(f"\nTarget found {result.occurrences}x, weakest match {result.weakest:.3f}; "
          f"best score elsewhere {result.background:.3f} (margin {result.margin:.3f})")
    for note in result.notes:
        print(f"Note: {note}")
    if result.suggested_tau is not None:
        print(f"\nSuggested: tau = {result.suggested_tau:.2f}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline

    cfg = load_config(args.config)
    a = cfg.audio
    pipe = Pipeline.from_config(cfg)
    chunk = a.period_size
    total_in = 0
    written = 0

    def step(frames: np.ndarray, writer: WavWriter) -> None:
        nonlocal written
        pipe.capture(frames)
        pipe.analyze(frames)
        out, pos = pipe.render(len(frames))
        if pos is not None and written < total_in:
            out = out[: total_in - written]
            writer.write(out)
            written += len(out)

    with WavWriter(args.output, a.sample_rate, a.channels) as writer:
        for frames in decode_chunks(args.input, a.sample_rate, a.channels, chunk):
            total_in += len(frames)
            step(frames, writer)
        silence = np.zeros((chunk, a.channels), np.float32)
        while written < total_in:  # flush the delay line
            step(silence, writer)

    print(f"Wrote {args.output} ({format_time(written / a.sample_rate)}), aligned with the input.")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    cmd = ["arecord", "-D", args.device, "-r", str(args.rate), "-c", str(args.channels),
           "-f", "S32_LE", "-d", str(args.duration), args.output]
    return subprocess.run(cmd).returncode


def cmd_run(args: argparse.Namespace) -> int:
    from .watchdog import Daemon

    daemon = Daemon(args.config, control=args.control)
    daemon.pipeline.set_mode(args.mode)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: daemon.stop_event.set())
    return daemon.run_forever()


def cmd_bench(args: argparse.Namespace) -> int:
    """Run the live pipeline on synthetic audio, as the daemon does (capture and playback per
    ALSA period, analysis per block), and time each stage. No audio devices needed."""
    import time

    from .config import AudioConfig
    from .features import NUM_MELS
    from .pipeline import ANALYSIS_BLOCK_S, Pipeline

    rate, period = args.rate, 1024
    rng = np.random.default_rng(0)
    audio = rng.normal(0.0, 0.1, (rate * args.seconds, 2))
    width = round(args.head / HOP_S)
    targets = [HeadTemplate(f"t{i}", f"t{i}", rng.normal(size=(width, NUM_MELS)), 0, 60.0, tau=2.0)
               for i in range(args.targets)]  # tau > 1: never fires, every frame is fully scored
    # Delay just above the recognition latency: realistic playback, no latency warnings.
    pipe = Pipeline(targets, {}, AudioConfig(sample_rate=rate, delay_s=args.head + 1.0, period_size=period))
    block = round(ANALYSIS_BLOCK_S * rate)

    spent = dict.fromkeys(("capture", "decimation", "features", "detection", "playback"), 0.0)
    clock = time.perf_counter
    pending = []
    for i in range(0, len(audio) - period + 1, period):
        chunk = audio[i : i + period]
        t0 = clock()
        pipe.capture(chunk)
        spent["capture"] += clock() - t0
        pending.append(chunk)
        if sum(len(p) for p in pending) >= block:
            frames = np.concatenate(pending)
            pending = []
            t0 = clock()
            x16k = pipe.decimator.process(frames.mean(axis=1))
            t1 = clock()
            feats, rms = pipe.extractor.process(x16k)
            t2 = clock()
            pipe.detect(feats, rms)
            t3 = clock()
            spent["decimation"] += t1 - t0
            spent["features"] += t2 - t1
            spent["detection"] += t3 - t2
        t0 = clock()
        pipe.render(period)
        spent["playback"] += clock() - t0

    fps = 1 / HOP_S
    comparisons = args.targets * fps
    stages = " + ".join(f"{len(t)}-tap FIR at {r // 1000} kHz" for t, r in pipe.decimator.stages())
    print(f"{args.seconds} s of {rate} Hz stereo, {args.targets} target(s) with a {width * HOP_S:.1f} s head\n")
    print(f"{'stage':<12}{'work per second of audio':<52}{'% of one core':>14}")
    work = {
        "capture": f"{rate // period} writes of {period} frames to the delay line",
        "decimation": f"downmix + {stages}",
        "features": f"{fps:.0f} FFTs of 2048 + {fps:.0f} x 60-band mel",
        "detection": f"{comparisons:.0f} comparisons x {width * NUM_MELS} MAC",
        "playback": f"{rate // period} reads of {period} frames",
    }
    for key, dt in spent.items():
        print(f"{key:<12}{work[key]:<52}{100 * dt / args.seconds:>13.2f}%")
    print(f"{'total':<64}{100 * sum(spent.values()) / args.seconds:>13.2f}%")
    return 0


def cmd_mopidy_output(args: argparse.Namespace) -> int:
    """Print Mopidy's `output =` value for the loopback (install-pi.sh does the same)."""
    from .mopidy import caps_rate, loopback_output

    rate = args.rate or caps_rate(args.original) or DEFAULT_SAMPLE_RATE
    print(rate if args.detect_rate else loopback_output(args.original, rate, args.alsa_latency_ms))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    from .library import Library

    config = Path(args.config).expanduser()
    if config.exists() and not args.force:
        print(f"{config} already exists (use --force to overwrite).")
        return 1
    library = Path(args.library).expanduser().resolve()
    text = example_config()
    values = {
        "library": str(library),
        "input_device": args.input_device,
        "output_device": args.output_device,
        "sample_rate": args.rate,
        "delay_s": args.delay,
    }
    for key, value in values.items():
        if value is not None:  # replace the value after `key = `, keeping the line's comment
            value = json.dumps(value)  # a valid TOML string or number
            text, n = re.subn(rf'^({key} = )("[^"]*"|[0-9.]+)', lambda m, v=value: m[1] + v, text, flags=re.M)
            if n != 1:
                raise RuntimeError(f"example config has no single `{key} =` line")

    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(text, encoding="utf-8")
    Library.open(library, args.rate or DEFAULT_SAMPLE_RATE)
    print(f"Wrote {config}")
    print(f"Library: {library}")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from .webserver import App, make_server

    if not args.studio and not args.onair:
        print("Error: enable at least one section: --studio [LIBRARY_DIR] and/or --onair [CONFIG]")
        return 2
    app = App(Path(args.studio) if args.studio else None, Path(args.onair) if args.onair else None,
              sample_rate=args.rate, autostart=not args.no_autostart)
    server = make_server(app, args.host, args.port)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: threading.Thread(target=server.shutdown).start())
    host, port = server.server_address[:2]
    logger.info("Web interface on http://%s:%d/", host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        app.close()
    return 0


def add_rate(p: argparse.ArgumentParser, help: str, default: int | None = DEFAULT_SAMPLE_RATE) -> None:
    p.add_argument("--rate", type=int, choices=SAMPLE_RATES, default=default, help=help)


def main() -> None:
    parser = argparse.ArgumentParser(prog="streamjanitor", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("precompute", help="(PC) target audio -> head template")
    p.add_argument("-i", "--input", required=True, help="target audio (mp3, wav, ...)")
    p.add_argument("-o", "--output", required=True, help="output .npz")
    p.add_argument("--start", type=float, default=0.0, help="trim: target starts at this offset (s)")
    p.add_argument("--duration", type=float, default=0.0, help="trim: target duration (s, 0 = to the end)")
    p.add_argument("--head", type=float, default=DEFAULT_HEAD_S, help="seconds of the target used for recognition")
    p.add_argument("--head-offset", default="auto",
                   help="where the head starts, seconds from the target start ('auto' = skip leading silence)")
    add_rate(p, "pipeline sample rate (must match the config)")
    p.add_argument("--gate", type=float, default=DEFAULT_ENERGY_GATE, help="silence RMS gate")
    p.add_argument("--name")
    p.add_argument("--id")
    p.set_defaults(func=cmd_precompute)

    p = sub.add_parser("calibrate", help="(PC) scan a recording with a template and suggest tau")
    p.add_argument("-i", "--input", required=True, help="long recording, e.g. the archived episode")
    p.add_argument("-t", "--target", required=True, help="template .npz")
    add_rate(p, "pipeline sample rate (must match the template's library)")
    p.add_argument("--gate", type=float, default=DEFAULT_ENERGY_GATE)
    p.add_argument("--min-voiced", type=float, default=DEFAULT_MIN_VOICED)
    p.add_argument("--top", type=int, default=8, help="how many matches to list")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("simulate", help="(PC) run the live pipeline on a file, write the output")
    p.add_argument("-c", "--config", default=str(default_config()))
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True, help="output .wav")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("record", help="(Pi) record the loopback to WAV")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-d", "--duration", type=int, default=1800, help="seconds")
    p.add_argument("-D", "--device", default="plughw:CARD=Loopback,DEV=1")
    add_rate(p, "sample rate (must match the config)")
    p.add_argument("-c", "--channels", type=int, default=2)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("run", help="(Pi) live daemon")
    p.add_argument("-c", "--config", default=str(default_config()))
    p.add_argument("--mode", choices=MODES, default="replace",
                   help="replace targets (default), report only (recognise, never cut), or passthrough "
                        "(no recognition: output = input)")
    p.add_argument("--control", action="store_true", help="JSON control on stdin/stdout (used by `web`)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("web", help="web interface (Studio and/or On Air)")
    p.add_argument("--host", default="127.0.0.1", help="listen address (0.0.0.0 = all interfaces)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--studio", metavar="LIBRARY_DIR", nargs="?", const=str(default_studio()),
                   help=f"enable Studio on this library folder (default {default_studio()})")
    p.add_argument("--onair", metavar="CONFIG", nargs="?", const=str(default_config()),
                   help=f"enable On Air: supervise the daemon of this config (default {default_config()})")
    add_rate(p, "sample rate for a new Studio library")
    p.add_argument("--no-autostart", action="store_true", help="don't start the daemon with the server")
    p.set_defaults(func=cmd_web)

    p = sub.add_parser("init", help="write a config file and an empty library in the standard locations")
    p.add_argument("-c", "--config", default=str(default_config()), help="where to write the config")
    p.add_argument("--library", default=str(default_library()), help="library folder for On Air")
    p.add_argument("--output-device", help="ALSA device of the DAC/HAT, e.g. plughw:CARD=sndrpihifiberry,DEV=0")
    p.add_argument("--input-device", help="ALSA capture device (default: the loopback)")
    p.add_argument("--delay", type=float, help="delay_s")
    add_rate(p, f"sample_rate (default {DEFAULT_SAMPLE_RATE})", default=None)
    p.add_argument("--force", action="store_true", help="overwrite an existing config")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("bench", help="measure the CPU cost of the live pipeline (no audio devices needed)")
    p.add_argument("-n", "--targets", type=int, default=10, help="number of active targets")
    p.add_argument("--head", type=float, default=DEFAULT_HEAD_S, help="head length in seconds")
    p.add_argument("--seconds", type=int, default=60, help="seconds of audio to process")
    add_rate(p, "pipeline sample rate")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("mopidy-output", help="print Mopidy's [audio] output for the loopback, keeping its chain")
    p.add_argument("--original", help="the current `output =` value, whose chain is kept")
    add_rate(p, f"default: the rate in the original's caps, or {DEFAULT_SAMPLE_RATE}", default=None)
    p.add_argument("--alsa-latency-ms", type=int, default=100, help="must match the config's alsa_latency_ms")
    p.add_argument("--detect-rate", action="store_true", help="only print the rate that would be used")
    p.set_defaults(func=cmd_mopidy_output)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(args.func(args))

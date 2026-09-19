# StreamJanitor

Once upon a time, there was this person who had resurrected an old raspberry, acquired a cheap s/pdif hat, and frankensteined a mopidy server to drive a yamaha a-s301.
That person liked lots of things, including whiskers on kittens, girls in white dresses, coffee, stories, but also pink floyd and classical music.
So Italy's Rai Radio 3 was playing pretty much constantly.

Then, as it happens in life, bad times came and that person started getting distressed by certain songs, jingles, or even names. And, misfortunes being really sociable things that happily cluster together, Radio 3 started playing a lot of that distressing stuff.

But!

Our person had a bit o' the knowing of the bits, and so he sat and pondered and eventually said "I shall find a way to separate what does not give me joy from what does give me joy, and then I will find peace" and got to work.

And the thing he did kinda worked, but he did not find peace.

Then he sat and pondered and eventually said "I shall try and make this thing more portable, and useful to other people. And in being useful to other people I will find happiness" and got to work. He also acquired a clanker subscription[^1] so that work wasn't really heavy.

This is the result. Alas, it will not bring you peace nor happiness, but maybe, just maybe, it will afford you a little solace from that annoying ad.

[^1]: They are useful tools, provided that you treat them as such. Also, life is too short to write tests, and I cannot do any kind of frontend so here you go.




Replaces known clips (songs, ads, segments) in a live stream outputted by Mopidy on a Raspberry Pi 3 (but it can adapted to anything that output audio).
Mopidy plays into an ALSA loopback; StreamJanitor listens to it, delays the audio by a few
seconds, and sends it to the S/PDIF HAT. When the opening seconds of a known target go by,
the target is _excised with extreme prejudice_, and replaced with a clip you choose
or with silence.

```
Mopidy ──▶ hw:Loopback,0 ═ loopback ═ hw:Loopback,1 ──▶ streamjanitor ──▶ S/PDIF HAT
                                                          │
                              delay line (delay_s) ───────┤ cut [start, end) + crossfades
                              16 kHz log-mel ─▶ head match┘
```

## How recognition works

- **Template ("head")**: `precompute` keeps only the first few seconds of the target (default 3 s,
  after any leading silence) as log-mel features[^2], plus the target's total duration. Size: a few KB.
- **Live**: every 20 ms the last 3 s of the stream are compared with each head (mean cosine
  similarity of the frames). The score peaks sharply when the stream lines up with the head, and
  the peak position gives the exact sample where the target started. The end is start + duration.
- **Delay**: the target is recognised about `head_offset + head + 0.2 s` after it starts. If
  `delay_s` is at least that, the cut starts before the target reaches the speakers (zero leak).
  If it's shorter, the difference leaks and the cut still ends on time.
- **Only the start is recognised**: if you tune in mid-target, or the start is covered (e.g. by
  the presenter talking), the target goes through. While a target is playing its detector is
  muted, so a song that repeats its intro cannot re-trigger.
- PC tools and the device run **the same feature code**, decimator included, so templates match
  the live features.

[^2]: basically, take the sound, get its spectrogram, chop it into buckets, using mel bands because it is something made for human hearing so that should be the best fitting model, bring out the numbers and voila.

## Audio quality

Outside a cut, the samples that reach the DAC are **bit for bit** the ones Mopidy produced:

- **Mopidy's chain is yours.** The installer keeps your `output` chain (converters,
  resampler, sample rate, format) and only changes the sink: the loopback instead of the HAT,
  plus the ALSA buffer and period sizes, which affect timing, not samples.
- **The loopback** copies samples unchanged.
- **streamjanitor** keeps them as float64, which holds any 32-bit sample exactly, and
  writes them back unchanged.
  `tests/test_audio_io.py` checks this with full 32-bit samples at 96 kHz.
- **Towards the DAC**, `plughw` converts the container format (e.g. 32-bit to the HAT's
  24-bit) exactly as it would if Mopidy played to the HAT directly.
- **The downsampling to 16 kHz** is only a copy for recognition; it never reaches the output.

What does change the audio:

- **Cuts**: the replacement (or silence), with equal-power crossfades at both ends.
  Replacements are stored as 24-bit WAV at the chain's rate, resampled with SoX if their
  file has another rate, and read from disk while playing.
- **Clock drift correction**: skips or repeats one sample when capture and playback drift
  more than 150 ms apart. With the loopback clocked by the HAT (`timer_source`) the two
  never drift, so it never happens. The On Air page counts it ("Drift corrections"):
  anything but 0 means the loopback isn't following the HAT.
- **Faults**: silence while the delay line refills after an underrun (counted too).

The whole chain runs at one sample rate: `sample_rate` in the config, the rate your Mopidy
chain forces, and the rate of the library (96 kHz by default; 48 kHz is also supported).

## CPU cost (cause I _love_ this stuff)

Work per second of audio at 96 kHz, with *n* active targets and the default 3 s head
(W = 150 frames of 60 mel bands). Everything but the delay line runs on the analysis copy,
in blocks of 100 ms:

| Stage | Work per second | Depends on |
| --- | --- | --- |
| Capture / playback: delay line | ~94 writes and reads of 1024 frames | fixed |
| Decimation: downmix, 96 → 48 kHz (11-tap FIR), 48 → 16 kHz (145 taps) | ≈ 1.1 + 7 M MAC | fixed (one step at 48 kHz) |
| Features: log-mel | 50 FFTs of 2048 points + 50 mel products ≈ 6 MFLOP | fixed |
| Detection: all heads against the last W frames, one matrix product per frame | 50 × 9,000·*n* MAC | *n* × head length |

The arithmetic is small; on a Pi what costs is the Python and numpy overhead around each
call. That's why the analysis works in 100 ms blocks and scores all targets in one matrix
product per frame instead of one small product per target. `streamjanitor bench` runs the
whole live pipeline on synthetic audio, as the daemon does (no audio devices needed), and
reports the share of one CPU core per stage:

```bash
streamjanitor bench -n 20          # 20 targets; --head, --rate, --seconds also available
```

Measured, % of one core at 96 kHz:

| Targets | 0 | 10 | 20 | 50 |
| --- | --- | --- | --- | --- |
| Desktop (Ryzen 7 7800X3D) | 0.38 | 0.43 | 0.46 | 0.52 |
| **Raspberry Pi 3** | | | **21.6** | |

On the Pi 3, with 20 targets: decimation 6.5%, features 4.9%, detection 8.5%, delay line
(capture + playback) 1.8%. The daemon is one Python process, so it effectively uses one of
the Pi 3's four cores; Mopidy runs on the others.

## Installation

StreamJanitor is installed as a [uv](https://docs.astral.sh/uv/) tool, from this repository
(a git URL or a local checkout). uv brings its own Python, so the system's doesn't matter.

### PC (Studio)

Needs `ffmpeg` (`sudo apt install ffmpeg`, `brew install ffmpeg`, ...), then:

```bash
uv tool install git+https://github.com/trynomial/streamjanitor
streamjanitor web --studio
```

and open http://127.0.0.1:8080/. The Studio library lives in `~/.local/share/streamjanitor/studio`
(or give a folder: `--studio ~/my-library`).

### Raspberry Pi (On Air)

Raspberry Pi OS **64-bit** (numpy has ready-made wheels only for aarch64), Mopidy already
playing the stream on the S/PDIF HAT. Get the repository on the Pi and run the installer as
your normal user:

```bash
git clone https://github.com/trynomial/streamjanitor && streamjanitor/deploy/install-pi.sh
```

It installs uv and streamjanitor, asks which sound card is the HAT, sets up the ALSA loopback
clocked by the HAT, writes `~/.config/streamjanitor/config.toml`, installs and enables the
`streamjanitor` systemd service (web interface on port 8084, whole LAN), and points Mopidy at
the loopback (asking first, with a backup of `mopidy.conf`), keeping your own output chain and
its sample rate (`--rate 96000|48000` to choose). 

```
Mopidy ──▶ loopback ──▶ streamjanitor (delay, cuts) ──▶ S/PDIF HAT
```

Open `http://<pi>.local:8084/`, and in **On Air** import the library exported from the Studio.
Options: `--dry-run` shows what would change, `--hat ID`, `--port N`, `--host ADDR`,
`--rate`, `--source git+https://...` (install from git instead of the checkout), `--yes`, `--help`.
Running it again keeps the installed port and address.

- **Update**: `git pull` and run the installer again (config and library are kept).
- **Uninstall**: `deploy/install-pi.sh --uninstall` removes the service, the loopback and
  the program, and restores Mopidy's config; reboot afterwards.
- **Logs**: the On Air page, or `journalctl -u streamjanitor -f`.

The loopback is clocked by the HAT (`timer_source`), so every device on it must use the HAT's
ALSA period: 25 ms (`alsa_latency_ms` / 4). That's why Mopidy's output ends with
`latency-time=25000`; if you change `alsa_latency_ms`, change it too. A mismatch shows up
in `dmesg` as *"Period size ... is not corresponding to timer resolution"* and as
`Input/output error` when streamjanitor reads the loopback.

Manual setup, if you prefer: `deploy/snd-aloop.conf` (loopback), `deploy/mopidy-audio.conf`
(Mopidy output), `streamjanitor init --output-device plughw:CARD=<hat>,DEV=0` (config), and
`deploy/streamjanitor.service` (fill in the `@...@` placeholders).

### Files

| What | Where |
| --- | --- |
| Config (devices, delay, library path) | `~/.config/streamjanitor/config.toml` |
| On Air library | `~/.local/share/streamjanitor/library/` |
| Studio library | `~/.local/share/streamjanitor/studio/` |

## Web interface

```bash
streamjanitor web --studio                    # Studio on the default library
streamjanitor web --onair --host 0.0.0.0      # On Air with the default config (what the Pi service runs)
streamjanitor web --studio ~/lib --onair ~/pi.toml --port 9000   # explicit paths, both sections
```

Options: `--host` (default `127.0.0.1`; `0.0.0.0`), `--port`
(default 8084). There is no password: expose it only on your home network.

**Studio** edits a *library* folder (created if missing):
- new pair from a ready-cut target, or from a whole episode plus the start and end times
  of the target (`h:mm:ss.s`);
- optional replacement: converted to 24-bit WAV at the library's rate (SoX resampling), and if longer than the
  target, cut to the target's duration with a 1 s fade-out; without one, silence;
- τ, the match threshold (0–1), is how closely the live audio must match the target's opening
  seconds before it is cut: higher means fewer false cuts but more missed targets;
- edit name, τ, head length and head start (the template is rebuilt from the kept source
  audio), change or remove the replacement, delete the pair, listen to target and replacement;
- **calibrate**: upload a whole episode and scan it with one pair's template (or all of them):
  lists where the target airs and how well it matches, the best match anywhere else, and a
  suggested τ between the two, with a button to apply it. Pick an episode in which the target
  airs, ideally not the one it was cut from (that one scores an unrealistic ~1.0);
- export as zip ("complete" keeps the source audio for later edits; "for On Air" only
  has what the Pi needs), or import a zip/tar (replaces the library; the old one stays as `.bak`).

**On Air** supervises the live daemon (started with the web server, restarted if it crashes):
- status, uptime, buffer and xrun counters, last detections, log;
- **Replace / Report only / Pass-through**: *replace* cuts recognised targets; *report only*
  keeps recognising and lists the detections, but never cuts; *pass-through* switches
  recognition off and copies the input to the output sample for sample (still delayed by
  `delay_s`, so switching mode never makes the audio jump). When recognition resumes after
  pass-through it needs a head's worth of audio (3 s by default) before it can detect again.
  The mode is remembered across restarts. **Stop** really stops the daemon, and with it the
  audio;
- enable/disable each pair and import/export the library: changes are applied on the fly,
  without interrupting the audio.

Library folder layout (`library.toml`, `targets/*.npz`, `replacements/*.wav`, `sources/*`):
`config.toml` points at it with `library = "..."`. Devices and delay stay in `config.toml`,
so importing a library never touches them.

## Command line

The same operations are available without the web interface. On the PC (needs `ffmpeg`):

```bash
# 1. template from the clip you cut from the archived episode
streamjanitor precompute -i target.mp3 -o targets/ad.npz --name "Ad"
#    options: --head 3.0 (seconds used for recognition), --head-offset auto|SECONDS,
#             --start/--duration to trim, --rate 96000 (must match config)

# 2. check it against the whole archived episode: lists every match, finds
#    re-airings, suggests tau from the gap between the target and everything else
streamjanitor calibrate -i episode.mp3 -t targets/ad.npz

# 3. listen to what the Pi would play (same pipeline, offline)
streamjanitor simulate -i episode.mp3 -o check.wav          # uses the default config; -c to pick one
```

If `calibrate` reports a small margin, try a longer `--head` (more delay needed), or a
`--head-offset` past a part the target shares with other clips (e.g. a common jingle at the
start of every ad).

## Replacement behaviour

The clip starts at the target's first sample. If it is shorter than the target the rest is
silence; if it is longer it is cut (with a fade) at the target's end. Both edges are
equal-power crossfades (`crossfade_ms`).

## Tests

```bash
uv run python -m unittest discover -s tests -t .
```

`tests/test_pipeline.py` runs the whole chain on synthetic audio and checks every output sample.

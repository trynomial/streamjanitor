"""The library: a folder of target/replacement pairs, the unit exchanged between Studio and On Air.

    library.toml          pairs and their settings
    targets/<id>.npz      head templates (what the Pi needs)
    replacements/<id>.wav replacement clips, ready to play: pipeline rate, 24 bit, stereo,
                          trimmed to the target duration (resampled with SoX if needed)
    sources/<id>.<ext>    target audio, kept to rebuild the template (Studio only)

Device and delay settings are not part of it: they belong to the machine (config.toml).
"""

import json
import re
import shutil
import tarfile
import tempfile
import tomllib
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath

import numpy as np

from .audiofile import WavWriter, decode, write_flac
from .config import DEFAULT_SAMPLE_RATE
from .detector import DEFAULT_TAU, HeadTemplate
from .precompute import DEFAULT_HEAD_S, make_template

LIBRARY_FILE = "library.toml"
LIBRARY_VERSION = 1
HEAD_RANGE_S = (0.5, 30.0)
TRIM_FADE_S = 1.0  # fade-out when a replacement is cut to the target length
END_FADE_S = 0.02
MAX_ARCHIVE_BYTES = 4 * 1024**3


class LibraryError(ValueError):
    pass


@dataclass
class Pair:
    id: str
    name: str
    duration_s: float
    target: str  # relative paths inside the library
    enabled: bool = True
    tau: float = DEFAULT_TAU
    head_s: float = DEFAULT_HEAD_S
    head_offset: str | float = "auto"  # "auto" = skip leading silence, else seconds
    source: str | None = None
    source_label: str | None = None
    replacement: str | None = None
    replacement_label: str | None = None

    def head_offset_s(self) -> float | None:
        return None if self.head_offset == "auto" else float(self.head_offset)


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:40] or "target"


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v))  # JSON string escapes are valid TOML basic-string escapes


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _inside(relative: str) -> bool:
    p = PurePosixPath(relative)
    return not p.is_absolute() and ".." not in p.parts


class Library:
    def __init__(self, root: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE, pairs: list[Pair] | None = None) -> None:
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.pairs = pairs or []

    # --- persistence ---

    @classmethod
    def load(cls, root: str | Path) -> "Library":
        root = Path(root)
        path = root / LIBRARY_FILE
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise LibraryError(f"{path} not found") from None
        except tomllib.TOMLDecodeError as e:
            raise LibraryError(f"{path}: {e}") from None
        if data.get("version") != LIBRARY_VERSION:
            raise LibraryError(f"{path}: unsupported version {data.get('version')!r}")
        known = {f.name for f in fields(Pair)}
        pairs = []
        for i, item in enumerate(data.get("pairs", [])):
            unknown = set(item) - known
            if unknown:
                raise LibraryError(f"{path}: pairs[{i}] has unknown keys {sorted(unknown)}")
            try:
                pairs.append(Pair(**item))
            except TypeError as e:
                raise LibraryError(f"{path}: pairs[{i}]: {e}") from None
        return cls(root, int(data.get("sample_rate", DEFAULT_SAMPLE_RATE)), pairs)

    @classmethod
    def open(cls, root: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> "Library":
        """Load, or create an empty library if the folder has none."""
        root = Path(root)
        if (root / LIBRARY_FILE).exists():
            return cls.load(root)
        lib = cls(root, sample_rate)
        lib.save()
        return lib

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        lines = [
            "# StreamJanitor library: target/replacement pairs (managed by the web Studio).",
            f"version = {LIBRARY_VERSION}",
            f"sample_rate = {self.sample_rate}",
        ]
        for pair in self.pairs:
            lines += ["", "[[pairs]]"]
            lines += [f"{k} = {_toml_value(v)}" for k, v in asdict(pair).items() if v is not None]
        _atomic_write(self.root / LIBRARY_FILE, "\n".join(lines) + "\n")

    # --- queries ---

    def get(self, pair_id: str) -> Pair:
        for pair in self.pairs:
            if pair.id == pair_id:
                return pair
        raise LibraryError(f"no pair with id '{pair_id}'")

    def path(self, relative: str) -> Path:
        return self.root / relative

    def validate(self) -> None:
        """Check every referenced file exists and every template loads."""
        ids = [p.id for p in self.pairs]
        if len(ids) != len(set(ids)):
            raise LibraryError("duplicate pair ids")
        for pair in self.pairs:
            for rel in (pair.target, pair.source, pair.replacement):
                if rel is not None and not _inside(rel):
                    raise LibraryError(f"pair '{pair.id}': path {rel!r} leaves the library")
            if pair.replacement and not self.path(pair.replacement).exists():
                raise LibraryError(f"pair '{pair.id}': missing {pair.replacement}")
            try:
                template = HeadTemplate.load(self.path(pair.target))
            except (OSError, ValueError, KeyError) as e:
                raise LibraryError(f"pair '{pair.id}': bad template {pair.target}: {e}") from None
            if template.target_id != pair.id:
                raise LibraryError(f"pair '{pair.id}': template belongs to '{template.target_id}'")

    # --- mutations (callers serialize access) ---

    def _new_id(self, name: str) -> str:
        base = slugify(name)
        taken = {p.id for p in self.pairs}
        candidate, n = base, 2
        while candidate in taken:
            candidate, n = f"{base}-{n}", n + 1
        return candidate

    def add_pair(
        self,
        name: str,
        audio_path: Path,
        audio_label: str,
        start_s: float | None = None,
        end_s: float | None = None,
        tau: float = DEFAULT_TAU,
        head_s: float = DEFAULT_HEAD_S,
        head_offset: str | float = "auto",
    ) -> tuple[Pair, list[str]]:
        """New pair from a ready-cut target (start_s None) or from [start_s, end_s) of an episode."""
        name = name.strip() or Path(audio_label).stem
        pair_id = self._new_id(name)
        (self.root / "sources").mkdir(parents=True, exist_ok=True)

        if start_s is None:
            mono = decode(audio_path, self.sample_rate, 1)
            suffix = Path(audio_label).suffix.lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,5}", suffix):
                suffix = ".audio"
            source = f"sources/{pair_id}{suffix}"
            shutil.copyfile(audio_path, self.path(source))
            label = audio_label
        else:
            if end_s is None or end_s <= start_s:
                raise LibraryError("the end must come after the start")
            mono = decode(audio_path, self.sample_rate, 1, start_s, end_s - start_s)
            if len(mono) < 0.9 * (end_s - start_s) * self.sample_rate:
                raise LibraryError(f"the episode is shorter than the requested end ({format_time(end_s)})")
            source = f"sources/{pair_id}.flac"
            write_flac(self.path(source), mono, self.sample_rate)
            label = f"{audio_label} {format_time(start_s)}–{format_time(end_s)}"

        pair = Pair(pair_id, name, round(len(mono) / self.sample_rate, 3), f"targets/{pair_id}.npz",
                    tau=tau, head_s=head_s, head_offset=head_offset, source=source, source_label=label)
        try:
            warnings = self._build_template(pair, mono[:, 0])
        except Exception:
            self.path(source).unlink(missing_ok=True)
            raise
        self.pairs.append(pair)
        self.save()
        return pair, warnings

    def _build_template(self, pair: Pair, mono: np.ndarray | None = None) -> list[str]:
        if mono is None:
            if not pair.source or not self.path(pair.source).exists():
                raise LibraryError(f"pair '{pair.id}' has no source audio: the template can't be rebuilt")
            mono = decode(self.path(pair.source), self.sample_rate, 1)[:, 0]
        try:
            template, warnings = make_template(mono, self.sample_rate, pair.id, pair.name,
                                               pair.head_s, pair.head_offset_s())
        except ValueError as e:
            raise LibraryError(str(e)) from None
        self.path(pair.target).parent.mkdir(parents=True, exist_ok=True)
        template.save(self.path(pair.target))
        return warnings

    def set_replacement(self, pair_id: str, audio_path: Path, audio_label: str) -> Pair:
        """Store a replacement ready to play: trimmed to the target, faded out, pipeline format."""
        pair = self.get(pair_id)
        audio = decode(audio_path, self.sample_rate, 2, hq=True)
        target_len = round(pair.duration_s * self.sample_rate)
        trimmed = len(audio) > target_len
        audio = audio[:target_len].copy()
        fade = min(len(audio), int((TRIM_FADE_S if trimmed else END_FADE_S) * self.sample_rate))
        if fade:
            audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]

        rel = f"replacements/{pair.id}.wav"
        self.path(rel).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path(rel + ".tmp")
        with WavWriter(tmp, self.sample_rate, 2, bits=24) as writer:
            writer.write(audio)
        tmp.replace(self.path(rel))
        pair.replacement = rel
        pair.replacement_label = audio_label + (" (trimmed to the target)" if trimmed else "")
        self.save()
        return pair

    def remove_replacement(self, pair_id: str) -> Pair:
        pair = self.get(pair_id)
        if pair.replacement:
            self.path(pair.replacement).unlink(missing_ok=True)
        pair.replacement = pair.replacement_label = None
        self.save()
        return pair

    def update(self, pair_id: str, **changes) -> tuple[Pair, list[str]]:
        """Change name/tau/enabled/head_s/head_offset; the template is rebuilt if the head changes."""
        pair = self.get(pair_id)
        convert = {
            "name": lambda v: str(v).strip() or pair.name,
            "tau": float,
            "enabled": bool,
            "head_s": float,
            "head_offset": lambda v: v if v == "auto" else float(v),
        }
        unknown = set(changes) - set(convert)
        if unknown:
            raise LibraryError(f"cannot change {sorted(unknown)}")
        changes = {key: convert[key](value) for key, value in changes.items()}
        if not 0.0 < changes.get("tau", pair.tau) <= 1.0:
            raise LibraryError("tau must be between 0 and 1")
        if not HEAD_RANGE_S[0] <= changes.get("head_s", pair.head_s) <= HEAD_RANGE_S[1]:
            raise LibraryError(f"the head must be between {HEAD_RANGE_S[0]} and {HEAD_RANGE_S[1]:.0f} s")

        old = asdict(pair)
        for key, value in changes.items():
            setattr(pair, key, value)

        warnings = []
        # The name lives in library.toml (it overrides the template's), so renaming needs no rebuild.
        if (pair.head_s, pair.head_offset) != (old["head_s"], old["head_offset"]):
            try:
                warnings = self._build_template(pair)
            except Exception:
                for key, value in old.items():
                    setattr(pair, key, value)
                raise
        self.save()
        return pair, warnings

    def delete(self, pair_id: str) -> None:
        pair = self.get(pair_id)
        for rel in (pair.target, pair.source, pair.replacement):
            if rel:
                self.path(rel).unlink(missing_ok=True)
        self.pairs.remove(pair)
        self.save()

    # --- archives ---

    def export_zip(self, dest, include_sources: bool = True) -> None:
        """Write the library as a zip to a path or binary file object."""
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.root / LIBRARY_FILE, LIBRARY_FILE)
            for pair in self.pairs:
                rels = [pair.target, pair.replacement] + ([pair.source] if include_sources else [])
                for rel in rels:
                    if rel and self.path(rel).exists():
                        zf.write(self.path(rel), rel)

    @classmethod
    def import_archive(cls, archive: Path, root: str | Path, sample_rate: int | None = None) -> "Library":
        """Replace the library at `root` with the one in a zip/tar archive (validated first).

        sample_rate: the rate the library will be played at; a library made for another rate
        is refused (its replacements would need resampling, and its templates re-checking).
        """
        root = Path(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".import-", dir=root.parent))
        try:
            _safe_extract(archive, staging)
            found = [p.parent for p in staging.rglob(LIBRARY_FILE)]
            if len(found) != 1:
                raise LibraryError(f"the archive must contain exactly one {LIBRARY_FILE} (found {len(found)})")
            imported = cls.load(found[0])
            imported.validate()
            if sample_rate and imported.pairs and imported.sample_rate != sample_rate:
                raise LibraryError(
                    f"this library is for {imported.sample_rate} Hz but this system runs at {sample_rate} Hz: "
                    f"export it from a {sample_rate} Hz Studio library (`streamjanitor web --studio DIR "
                    f"--rate {sample_rate}` creates one)"
                )

            backup = root.with_name(root.name + ".bak")
            if backup.exists():
                shutil.rmtree(backup)
            if root.exists():
                root.replace(backup)
            found[0].replace(root)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return cls.load(root)


def _check_entries(entries: list[tuple[str, int]]) -> None:
    """Archive entries (name, size): refuse paths leaving the destination, and zip bombs."""
    for name, _ in entries:
        if not _inside(name):
            raise LibraryError(f"unsafe path in archive: {name}")
    if sum(size for _, size in entries) > MAX_ARCHIVE_BYTES:
        raise LibraryError("archive too large")


def _safe_extract(archive: Path, dest: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            _check_entries([(i.filename, i.file_size) for i in zf.infolist()])
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            _check_entries([(m.name, m.size) for m in members])
            for m in members:
                if not (m.isfile() or m.isdir()):
                    raise LibraryError(f"unsupported entry in archive: {m.name}")
            if hasattr(tarfile, "data_filter"):  # Python >= 3.11.4
                tf.extractall(dest, members=members, filter="data")
            else:  # members were checked above: plain files and dirs, relative, no ".."
                tf.extractall(dest, members=members)
    else:
        raise LibraryError("not a zip or tar archive")


def format_time(seconds: float, decimals: int = 1) -> str:
    """3723.5 -> '1:02:03.5'; hours only when needed: 63.5 -> '1:03.5'."""
    h, rem = divmod(max(0.0, seconds), 3600)
    m, s = divmod(rem, 60)
    sec = f"{s:0{3 + decimals}.{decimals}f}"
    return f"{int(h)}:{int(m):02d}:{sec}" if h else f"{int(m)}:{sec}"


def parse_time(text: str) -> float:
    """'1:02:03.5', '62:03.5' or '3723.5' -> seconds."""
    parts = str(text).strip().replace(",", ".").split(":")
    if not 1 <= len(parts) <= 3 or not all(parts):
        raise LibraryError(f"invalid time {text!r}")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise LibraryError(f"invalid time {text!r}") from None
    seconds = 0.0
    for v in values:
        seconds = seconds * 60 + v
    if seconds < 0:
        raise LibraryError(f"invalid time {text!r}")
    return seconds

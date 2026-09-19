"""Web interface: Studio (edit a library) and On Air (run the live daemon).

Standard library only. The page (static/index.html) talks to a small JSON API;
uploads are raw request bodies (file name in X-Filename) streamed to a temp file.
Mutating requests must carry `X-Requested-With: streamjanitor`: browsers don't send
custom headers cross-origin without a CORS preflight (which is never granted), so
other web pages can't drive the server.
"""

import json
import logging
import mimetypes
import re
import tempfile
import threading
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .calibrate import calibrate, recording_features
from .config import DEFAULT_SAMPLE_RATE
from .detector import DEFAULT_TAU, HeadTemplate
from .features import HOP_S
from .library import Library, parse_time
from .onair import OnAir
from .precompute import DEFAULT_HEAD_S, detection_latency_s

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 4 * 1024**3
CHUNK = 1 << 20  # bytes per read/write when streaming bodies and files
CSRF_HEADER = ("X-Requested-With", "streamjanitor")


class HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class App:
    def __init__(
        self,
        studio_root: Path | None,
        onair_config: Path | None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        autostart: bool = True,
    ) -> None:
        self.lock = threading.Lock()  # serializes library changes
        self.studio_root = studio_root.resolve() if studio_root else None
        if self.studio_root:
            Library.open(self.studio_root, sample_rate)

        self.onair = OnAir(onair_config) if onair_config else None
        self.onair_root: Path | None = None
        self.onair_rate = DEFAULT_SAMPLE_RATE
        if onair_config:
            # Only the library path and rate are needed here; the daemon reads the rest.
            data = tomllib.loads(Path(onair_config).read_text(encoding="utf-8"))
            self.onair_rate = int(data.get("audio", {}).get("sample_rate", DEFAULT_SAMPLE_RATE))
            if "library" in data:
                self.onair_root = (Path(onair_config).parent / data["library"]).resolve()
                Library.open(self.onair_root, self.onair_rate)
        if self.onair and autostart:
            self.onair.start()

    def root(self, section: str) -> Path:
        root = self.studio_root if section == "studio" else self.onair_root
        if root is None:
            raise HttpError(HTTPStatus.NOT_FOUND, f"{section} has no library")
        return root

    def library(self, section: str) -> Library:
        return Library.load(self.root(section))

    @contextmanager
    def editing(self, section: str) -> Iterator[Library]:
        """Change a library under the lock; afterwards the live daemon reloads if it uses it."""
        with self.lock:
            yield self.library(section)
        self.changed(self.root(section))

    def changed(self, root: Path) -> None:
        """Library at `root` changed: tell the daemon if it's the live one."""
        if self.onair and self.onair_root == root:
            self.onair.reload()

    def close(self) -> None:
        if self.onair:
            self.onair.close()


def pair_json(lib: Library, pair) -> dict:
    data = asdict(pair)
    try:
        template = HeadTemplate.load(lib.path(pair.target))
        data["head_used_s"] = round(template.width * HOP_S, 2)
        data["head_offset_used_s"] = round(template.head_offset * HOP_S, 2)
        data["latency_s"] = round(detection_latency_s(template), 1)
    except (OSError, ValueError, KeyError):
        data["template_error"] = True
    return data


class Handler(BaseHTTPRequestHandler):
    app: App  # set on the subclass created by make_server()
    server_version = "streamjanitor"

    # --- plumbing ---

    def log_message(self, fmt, *args) -> None:
        logger.debug("%s %s", self.address_string(), fmt % args)

    def _dispatch(self, method: str) -> None:
        url = urlsplit(self.path)
        self.query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        self._body_left = int(self.headers.get("Content-Length") or 0)
        try:
            if method != "GET" and self.headers.get(CSRF_HEADER[0]) != CSRF_HEADER[1]:
                raise HttpError(HTTPStatus.FORBIDDEN, "missing X-Requested-With header")
            for pattern, route_method, fn in ROUTES:
                m = re.fullmatch(pattern, url.path)
                if m and route_method == method:
                    return fn(self, *m.groups())
            raise HttpError(HTTPStatus.NOT_FOUND, f"no route for {method} {url.path}")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            if isinstance(e, HttpError):
                status = e.status
            elif isinstance(e, (ValueError, RuntimeError, FileNotFoundError)):  # incl. LibraryError
                status = HTTPStatus.BAD_REQUEST
            else:
                logger.exception("Unhandled error on %s %s", method, url.path)
                status = HTTPStatus.INTERNAL_SERVER_ERROR
            self._discard_body()
            self._json({"error": str(e)}, status)

    def _read_body(self, n: int) -> bytes:
        data = self.rfile.read(min(n, self._body_left))
        self._body_left -= len(data)
        return data

    def _discard_body(self) -> None:
        """Read what's left of the request body: answering before the client finished
        sending makes browsers report a network error instead of our message."""
        if self._body_left > MAX_UPLOAD_BYTES:
            self.close_connection = True
            return
        while self._body_left > 0 and self._read_body(CHUNK):
            pass

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _json(self, data, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        try:
            data = json.loads(self._read_body(self._body_left) or b"{}")
        except json.JSONDecodeError as e:
            raise HttpError(HTTPStatus.BAD_REQUEST, f"bad JSON: {e}") from None
        if not isinstance(data, dict):
            raise HttpError(HTTPStatus.BAD_REQUEST, "expected a JSON object")
        return data

    def _receive_upload(self, tmpdir: str) -> tuple[Path, str]:
        """Stream the request body to a temp file; returns (path, original file name)."""
        if self._body_left <= 0:
            raise HttpError(HTTPStatus.BAD_REQUEST, "empty upload")
        if self._body_left > MAX_UPLOAD_BYTES:
            raise HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "upload too large")
        name = Path(unquote(self.headers.get("X-Filename", "upload"))).name or "upload"
        path = Path(tmpdir) / ("upload" + Path(name).suffix[:8])
        with path.open("wb") as f:
            while self._body_left:
                chunk = self._read_body(CHUNK)
                if not chunk:
                    raise HttpError(HTTPStatus.BAD_REQUEST, "upload interrupted")
                f.write(chunk)
        return path, name

    def _send_file(self, path: Path, content_type: str, download_name: str | None = None) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m[1] or m[2]):
            if m[1]:
                start, end = int(m[1]), min(int(m[2]) if m[2] else size - 1, size - 1)
            else:
                start = max(0, size - int(m[2]))
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = f.read(min(remaining, CHUNK))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # --- general ---

    def index(self):
        body = resources.files("streamjanitor").joinpath("static/index.html").read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")  # an update must not leave an old page talking to a new API
        self.end_headers()
        self.wfile.write(body)

    def info(self):
        app = self.app
        same = app.studio_root is not None and app.studio_root == app.onair_root
        self._json({
            "studio": app.studio_root is not None,
            "onair": app.onair is not None,
            "studio_library": str(app.studio_root) if app.studio_root else None,
            "onair_library": str(app.onair_root) if app.onair_root else None,
            "studio_rate": Library.load(app.studio_root).sample_rate if app.studio_root else None,
            "onair_rate": app.onair_rate if app.onair else None,
            "shared_library": same,
        })

    # --- library (both sections) ---

    def list_pairs(self, section: str):
        lib = self.app.library(section)
        self._json({"sample_rate": lib.sample_rate, "pairs": [pair_json(lib, p) for p in lib.pairs]})

    def export(self, section: str):
        lib = self.app.library(section)
        include_sources = self.query.get("sources", "1") != "0"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "library.zip"
            with self.app.lock:
                lib.export_zip(path, include_sources)
            suffix = "" if include_sources else "-live"
            self._send_file(path, "application/zip", f"streamjanitor-library{suffix}.zip")

    def import_(self, section: str):
        root = self.app.root(section)
        with tempfile.TemporaryDirectory() as tmp:
            archive, _ = self._receive_upload(tmp)
            with self.app.lock:
                lib = Library.import_archive(archive, root, self.app.onair_rate if section == "onair" else None)
        self.app.changed(root)
        self._json({"pairs": len(lib.pairs)})

    # --- Studio ---

    def studio_add(self):
        q = self.query
        start = parse_time(q["start"]) if q.get("start") else None
        end = parse_time(q["end"]) if q.get("end") else None
        if (start is None) != (end is None):
            raise HttpError(HTTPStatus.BAD_REQUEST, "give both start and end, or neither")
        with tempfile.TemporaryDirectory() as tmp:
            audio, filename = self._receive_upload(tmp)
            with self.app.editing("studio") as lib:
                pair, warnings = lib.add_pair(
                    q.get("name", ""), audio, filename, start, end,
                    tau=float(q.get("tau", DEFAULT_TAU)), head_s=float(q.get("head_s", DEFAULT_HEAD_S)),
                )
        self._json({"pair": pair_json(lib, pair), "warnings": warnings})

    def studio_update(self, pair_id: str):
        changes = self._read_json()
        with self.app.editing("studio") as lib:
            pair, warnings = lib.update(pair_id, **changes)
        self._json({"pair": pair_json(lib, pair), "warnings": warnings})

    def studio_delete(self, pair_id: str):
        with self.app.editing("studio") as lib:
            lib.delete(pair_id)
        self._json({"ok": True})

    def studio_set_replacement(self, pair_id: str):
        with tempfile.TemporaryDirectory() as tmp:
            audio, filename = self._receive_upload(tmp)
            with self.app.editing("studio") as lib:
                pair = lib.set_replacement(pair_id, audio, filename)
        self._json({"pair": pair_json(lib, pair)})

    def studio_remove_replacement(self, pair_id: str):
        with self.app.editing("studio") as lib:
            pair = lib.remove_replacement(pair_id)
        self._json({"pair": pair_json(lib, pair)})

    def studio_calibrate(self):
        """Scan an uploaded episode with one pair's template (or all of them)."""
        lib = self.app.library("studio")
        wanted = self.query.get("pair", "all")
        pairs = lib.pairs if wanted == "all" else [lib.get(wanted)]
        if not pairs:
            raise HttpError(HTTPStatus.BAD_REQUEST, "the library has no pairs")
        with tempfile.TemporaryDirectory() as tmp:
            episode, _ = self._receive_upload(tmp)
            # Read-only work: no library lock, so the page stays usable meanwhile.
            frames, rms = recording_features(episode, lib.sample_rate)
        results = []
        for pair in pairs:
            item = {"id": pair.id, "name": pair.name, "tau": pair.tau}
            try:
                result = calibrate(HeadTemplate.load(lib.path(pair.target)), frames, rms)
            except (OSError, ValueError, KeyError) as e:
                item["error"] = str(e)
            else:
                item.update(asdict(result), margin=round(result.margin, 3))
            results.append(item)
        self._json({"duration_s": round(len(frames) * HOP_S, 1), "results": results})

    def pair_audio(self, section: str, pair_id: str, which: str):
        lib = self.app.library(section)
        rel = getattr(lib.get(pair_id), "source" if which == "target" else "replacement")
        if not rel or not lib.path(rel).exists():
            raise HttpError(HTTPStatus.NOT_FOUND, f"no {which} audio for '{pair_id}'")
        path = lib.path(rel)
        self._send_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    # --- On Air ---

    def _onair(self) -> OnAir:
        if self.app.onair is None:
            raise HttpError(HTTPStatus.NOT_FOUND, "On Air is not enabled")
        return self.app.onair

    def onair_status(self):
        status = self._onair().status()
        status["library"] = None
        if self.app.onair_root is not None:
            lib = Library.load(self.app.onair_root)
            status["library"] = [{"id": p.id, "name": p.name, "enabled": p.enabled,
                                  "has_replacement": p.replacement is not None} for p in lib.pairs]
        self._json(status)

    def onair_action(self, action: str):
        onair = self._onair()
        if action == "mode":
            onair.set_mode(self._read_json().get("mode"))
        else:
            getattr(onair, action)()
        self._json(onair.status())

    def onair_toggle(self, pair_id: str):
        enabled = bool(self._read_json().get("enabled"))
        with self.app.editing("onair") as lib:
            lib.update(pair_id, enabled=enabled)
        self._json({"ok": True})


SECTION = r"(studio|onair)"
PAIR = r"([a-z0-9-]+)"
ROUTES = [
    (r"/", "GET", Handler.index),
    (r"/api/info", "GET", Handler.info),
    (rf"/api/{SECTION}/pairs", "GET", Handler.list_pairs),
    (rf"/api/{SECTION}/export", "GET", Handler.export),
    (rf"/api/{SECTION}/import", "POST", Handler.import_),
    (rf"/api/{SECTION}/pairs/{PAIR}/audio/(target|replacement)", "GET", Handler.pair_audio),
    (r"/api/studio/pairs", "POST", Handler.studio_add),
    (r"/api/studio/calibrate", "POST", Handler.studio_calibrate),
    (rf"/api/studio/pairs/{PAIR}", "PATCH", Handler.studio_update),
    (rf"/api/studio/pairs/{PAIR}", "DELETE", Handler.studio_delete),
    (rf"/api/studio/pairs/{PAIR}/replacement", "PUT", Handler.studio_set_replacement),
    (rf"/api/studio/pairs/{PAIR}/replacement", "DELETE", Handler.studio_remove_replacement),
    (r"/api/onair/status", "GET", Handler.onair_status),
    (r"/api/onair/(start|stop|restart|mode)", "POST", Handler.onair_action),
    (rf"/api/onair/pairs/{PAIR}", "PATCH", Handler.onair_toggle),
]


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server

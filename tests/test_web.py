"""Web API, with the server running in-process on a free port."""

import ctypes.util
import json
import shutil
import tempfile
import threading
import time
import unittest
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import HTTPError

import numpy as np

from streamjanitor.audiofile import WavWriter
from streamjanitor.onair import OnAir
from streamjanitor.webserver import App, make_server

from .synth import melody

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_ALSA = ctypes.util.find_library("asound") is not None
CSRF = {"X-Requested-With": "streamjanitor"}


class Client:
    def __init__(self, base):
        self.base = base

    def call(self, method, path, body=None, headers=None, raw=False):
        headers = dict(headers or {})
        data = body
        if isinstance(body, dict):
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                content = r.read()
                return r.status, (content if raw else json.loads(content)), r.headers
        except HTTPError as e:
            return e.code, json.loads(e.read() or b"{}"), e.headers

    def upload(self, method, path, file: Path):
        return self.call(method, path, file.read_bytes(), {**CSRF, "X-Filename": file.name})


class ServerCase(unittest.TestCase):
    studio = True
    onair = False
    output_device = "null"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        config = None
        if self.onair:
            config = self.tmp / "config.toml"
            config.write_text(f'library = "live"\n[audio]\ninput_device = "null"\n'
                              f'output_device = "{self.output_device}"\ndelay_s = 1.0\n')
        self.app = App(self.tmp / "studio" if self.studio else None, config)
        self.server = make_server(self.app, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.c = Client(f"http://127.0.0.1:{self.server.server_address[1]}")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.app.close()
        shutil.rmtree(self.tmp)

    def wav(self, name, seconds, seed):
        path = self.tmp / name
        with WavWriter(path, 48000, 2) as w:
            w.write(melody(seconds, seed))
        return path


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class TestStudioApi(ServerCase):
    def test_page_and_info(self):
        status, body, _ = self.c.call("GET", "/", raw=True)
        self.assertEqual(status, 200)
        self.assertIn(b"StreamJanitor", body)
        status, info, _ = self.c.call("GET", "/api/info")
        self.assertEqual((info["studio"], info["onair"]), (True, False))

    def test_mutations_need_csrf_header(self):
        status, _, _ = self.c.call("POST", "/api/studio/import", b"x")
        self.assertEqual(status, 403)

    def test_pair_lifecycle(self):
        episode = self.wav("episode.wav", 20.0, 2)
        status, r, _ = self.c.upload("POST", "/api/studio/pairs?name=Spot&start=0:05&end=0:13", episode)
        self.assertEqual(status, 200, r)
        pair = r["pair"]
        self.assertEqual((pair["id"], pair["duration_s"], pair["head_used_s"]), ("spot", 8.0, 3.0))

        status, r, _ = self.c.upload("PUT", "/api/studio/pairs/spot/replacement", self.wav("long.wav", 30.0, 3))
        self.assertIn("trimmed", r["pair"]["replacement_label"])

        status, r, _ = self.c.call("PATCH", "/api/studio/pairs/spot", {"tau": 0.72, "head_s": 2.0}, CSRF)
        self.assertEqual((r["pair"]["tau"], r["pair"]["head_used_s"]), (0.72, 2.0))
        status, r, _ = self.c.call("PATCH", "/api/studio/pairs/spot", {"tau": 7}, CSRF)
        self.assertEqual(status, 400)

        status, body, headers = self.c.call("GET", "/api/studio/pairs/spot/audio/replacement",
                                            headers={"Range": "bytes=0-99"}, raw=True)
        self.assertEqual((status, len(body)), (206, 100))
        self.assertTrue(headers["Content-Range"].startswith("bytes 0-99/"))

        status, body, _ = self.c.call("GET", "/api/studio/export?sources=0", raw=True)
        zip_path = self.tmp / "out.zip"
        zip_path.write_bytes(body)
        self.assertEqual(sorted(zipfile.ZipFile(zip_path).namelist()),
                         ["library.toml", "replacements/spot.wav", "targets/spot.npz"])

        status, r, _ = self.c.call("DELETE", "/api/studio/pairs/spot", headers=CSRF)
        self.assertEqual(status, 200)
        status, r, _ = self.c.call("GET", "/api/studio/pairs")
        self.assertEqual(r["pairs"], [])

        status, r, _ = self.c.upload("POST", "/api/studio/import", zip_path)
        self.assertEqual(r, {"pairs": 1})

    def test_calibrate(self):
        target = self.wav("t.wav", 6.0, 1)
        self.c.upload("POST", "/api/studio/pairs?name=Spot", target)
        episode = self.tmp / "episode.wav"
        with WavWriter(episode, 48000, 2) as w:
            w.write(np.concatenate((melody(15.0, 2), melody(6.0, 1), melody(60.0, 3))))
        status, r, _ = self.c.upload("POST", "/api/studio/calibrate?pair=spot", episode)
        self.assertEqual(status, 200, r)
        (res,) = r["results"]
        self.assertEqual((res["id"], res["occurrences"]), ("spot", 1))
        self.assertAlmostEqual(res["matches"][0]["start_s"], 15.0, delta=0.03)
        self.assertLess(res["background"], res["suggested_tau"])
        status, r, _ = self.c.upload("POST", "/api/studio/calibrate?pair=nope", episode)
        self.assertEqual(status, 400)

    def test_bad_requests_are_reported(self):
        clip = self.wav("t.wav", 5.0, 1)
        for path in ("/api/studio/pairs?start=0:05", "/api/studio/pairs?start=x&end=y"):
            status, r, _ = self.c.upload("POST", path, clip)
            self.assertEqual(status, 400)
            self.assertIn("error", r)
        status, r, _ = self.c.call("GET", "/api/studio/pairs/nope/audio/target")
        self.assertEqual(status, 400)
        status, r, _ = self.c.call("GET", "/api/onair/status")
        self.assertEqual(status, 404)


@unittest.skipUnless(HAVE_FFMPEG and HAVE_ALSA, "needs ffmpeg and libasound")
class TestOnAirApi(ServerCase):
    studio = False
    onair = True

    def wait_state(self, state, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            s = self.c.call("GET", "/api/onair/status")[1]
            if s["state"] == state:
                return s
            time.sleep(0.2)
        raise AssertionError(f"daemon never reached {state!r}: {s}")

    def test_daemon_control(self):
        s = self.wait_state("running")
        self.assertEqual(s["library"], [])

        self.assertEqual(self.c.call("POST", "/api/onair/mode", {"mode": "bogus"}, CSRF)[0], 400)
        self.c.call("POST", "/api/onair/mode", {"mode": "passthrough"}, CSRF)
        deadline = time.time() + 5
        while self.c.call("GET", "/api/onair/status")[1]["daemon"]["mode"] != "passthrough":
            self.assertLess(time.time(), deadline)
            time.sleep(0.2)

        self.c.call("POST", "/api/onair/stop", headers=CSRF)
        self.wait_state("stopped")
        time.sleep(4)  # the watchdog must not restart a daemon stopped on purpose
        self.assertEqual(self.c.call("GET", "/api/onair/status")[1]["state"], "stopped")
        self.c.call("POST", "/api/onair/start", headers=CSRF)
        s = self.wait_state("running")
        self.assertEqual(s["daemon"]["mode"], "passthrough", "the mode must survive a restart")


class TestOnAirState(unittest.TestCase):
    def mode_from(self, state: str | None) -> str:
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / "config.toml"
            if state is not None:
                config.with_name(".onair-state.json").write_text(state)
            onair = OnAir(config)
            onair.close()
            return onair.mode

    def test_saved_mode(self):
        self.assertEqual(self.mode_from(None), "replace")
        self.assertEqual(self.mode_from('{"mode": "passthrough"}'), "passthrough")
        self.assertEqual(self.mode_from('{"mode": "nonsense"}'), "replace")
        self.assertEqual(self.mode_from("[]"), "replace")

    def test_old_bypass_state_means_report_only(self):
        self.assertEqual(self.mode_from('{"bypass": true}'), "report")
        self.assertEqual(self.mode_from('{"bypass": false}'), "replace")


@unittest.skipUnless(HAVE_ALSA, "needs libasound")
class TestOnAirCrash(ServerCase):
    studio = False
    onair = True
    output_device = "hw:DoesNotExist"

    def test_exit_reason_is_reported(self):
        deadline = time.time() + 15
        while time.time() < deadline:
            status = self.c.call("GET", "/api/onair/status")[1]
            if status["last_exit"]:
                break
            time.sleep(0.2)
        self.assertIn("hw:DoesNotExist: open failed", status["last_exit"] or "")
        self.assertEqual(status["state"], "crashed")


if __name__ == "__main__":
    unittest.main()

"""On Air: supervises the live daemon as a child process (`streamjanitor run --control`).

The daemon runs in its own process so heavy work in the web server (e.g. Studio
decoding an episode) can't starve the audio threads. The supervisor restarts it
if it dies, forwards mode / reload commands, and keeps its last status and
log lines. The mode (replace, report only, pass-through) is persisted next to the
config file, so it survives reboots.
"""

import json
import logging
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from .config import MODES

logger = logging.getLogger(__name__)

RESTART_DELAY_S = 3.0
STATUS_STALE_S = 5.0


class OnAir:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.state_path = self.config_path.with_name(".onair-state.json")
        self.mode = self._load_mode()
        self.log: deque[str] = deque(maxlen=300)
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._want_running = False
        self._status: dict | None = None
        self._status_time = 0.0
        self.restarts = 0
        self.last_exit: str | None = None
        self._reported_pid: int | None = None
        self._stderr_reader: threading.Thread | None = None
        self._last_stderr = ""  # last line the daemon printed: usually the error it died of
        self._closed = False
        threading.Thread(target=self._watch, name="onair-watch", daemon=True).start()

    # --- persistent state ---

    def _load_mode(self) -> str:
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return "replace"
        if not isinstance(state, dict):
            return "replace"
        if "bypass" in state:  # older versions: "bypass" was what is now report only
            return "report" if state["bypass"] else "replace"
        return state.get("mode") if state.get("mode") in MODES else "replace"

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps({"mode": self.mode}))

    # --- process control ---

    def start(self) -> None:
        with self._lock:
            self._want_running = True
            if self._running():
                return
            cmd = [sys.executable, "-m", "streamjanitor", "run", "-c", str(self.config_path), "--control",
                   "--mode", self.mode]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, cwd=self.config_path.parent,
            )
            self._status, self._last_stderr = None, ""
            self._log(f"daemon started (pid {self._proc.pid})")
            threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
            self._stderr_reader = threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True)
            self._stderr_reader.start()

    def stop(self) -> None:
        with self._lock:
            self._want_running = False
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self._send({"cmd": "stop"})
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._log("daemon stopped")

    def restart(self) -> None:
        self.stop()
        self.start()

    def close(self) -> None:
        self._closed = True
        self.stop()

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r} (expected one of {', '.join(MODES)})")
        self.mode = mode
        self._save_state()
        self._send({"cmd": "mode", "mode": mode})

    def reload(self) -> None:
        self._send({"cmd": "reload"})

    def _running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _send(self, msg: dict) -> None:
        with self._lock:
            if not self._running():
                return
            try:
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    # --- background threads ---

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            try:
                status = json.loads(line)["status"]
            except (ValueError, KeyError, TypeError):
                continue
            with self._lock:
                if proc is self._proc:
                    self._status, self._status_time = status, time.time()

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        """The daemon's log goes to the page and to our stderr (the journal under systemd)."""
        for line in proc.stderr:
            line = line.rstrip()
            self.log.append(line)
            sys.stderr.write(f"daemon[{proc.pid}]: {line}\n")
            sys.stderr.flush()
            if line.strip():
                self._last_stderr = line.strip()

    def _watch(self) -> None:
        while not self._closed:
            time.sleep(0.5)
            with self._lock:
                proc, want, reader = self._proc, self._want_running, self._stderr_reader
                if proc is None or proc.poll() is None:
                    continue
                new_exit = self._reported_pid != proc.pid
                self._reported_pid = proc.pid
            if new_exit:
                if reader:
                    reader.join(timeout=2.0)  # let the last lines (usually the error) arrive
                self.last_exit = f"exit code {proc.returncode}" + (f": {self._last_stderr}" if self._last_stderr else "")
                if want:
                    self._log(f"daemon exited unexpectedly ({self.last_exit}); restarting in {RESTART_DELAY_S:.0f} s")
            if want:
                time.sleep(RESTART_DELAY_S)
                with self._lock:
                    if self._want_running and not self._running() and not self._closed:
                        self.restarts += 1
                        self.start()

    def _log(self, text: str) -> None:
        logger.info(text)
        self.log.append(time.strftime("%Y-%m-%d %H:%M:%S ") + "[on-air] " + text)

    # --- status ---

    def status(self) -> dict:
        with self._lock:
            running = self._running()
            fresh = self._status is not None and time.time() - self._status_time < STATUS_STALE_S
            if running:
                state = "running" if fresh else "starting"
            elif self._want_running:
                state = "crashed"
            else:
                state = "stopped"
            return {
                "state": state,
                "mode": self.mode,
                "pid": self._proc.pid if running else None,
                "restarts": self.restarts,
                "last_exit": self.last_exit,
                "daemon": self._status if (running and fresh) else None,
                "log": list(self.log)[-60:],
            }

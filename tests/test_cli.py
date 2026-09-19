import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from streamjanitor.config import load_config


class TestInit(unittest.TestCase):
    def run_cli(self, env, *args):
        return subprocess.run([sys.executable, "-m", "streamjanitor", *args], env=env,
                              capture_output=True, text=True)

    def test_init_writes_config_and_library_in_xdg_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "XDG_CONFIG_HOME": f"{d}/cfg", "XDG_DATA_HOME": f"{d}/data"}
            r = self.run_cli(env, "init", "--output-device", "plughw:CARD=hat,DEV=0", "--delay", "5")
            self.assertEqual(r.returncode, 0, r.stderr)
            config = Path(d) / "cfg/streamjanitor/config.toml"
            cfg = load_config(config)
            self.assertEqual(cfg.audio.output_device, "plughw:CARD=hat,DEV=0")
            self.assertEqual(cfg.audio.delay_s, 5.0)
            self.assertEqual(cfg.library, Path(d) / "data/streamjanitor/library")
            self.assertTrue((cfg.library / "library.toml").exists())

            r = self.run_cli(env, "init")
            self.assertEqual(r.returncode, 1, "must not overwrite without --force")
            self.assertEqual(load_config(config).audio.output_device, "plughw:CARD=hat,DEV=0")


if __name__ == "__main__":
    unittest.main()

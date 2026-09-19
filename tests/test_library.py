import io
import shutil
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from streamjanitor.audiofile import WavWriter, decode
from streamjanitor.config import load_config
from streamjanitor.detector import HeadTemplate
from streamjanitor.library import Library, LibraryError, parse_time

from .synth import melody

RATE = 48000
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def write_wav(path, audio):
    with WavWriter(path, RATE, 2) as w:
        w.write(audio)
    return path


class TestParseTime(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_time("75.5"), 75.5)
        self.assertEqual(parse_time("1:15,5"), 75.5)
        self.assertEqual(parse_time("1:01:15.5"), 3675.5)
        for bad in ("", "a:b", "1::2", "-3", "1:2:3:4"):
            with self.assertRaises(LibraryError):
                parse_time(bad)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class TestLibrary(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.target = melody(8.0, seed=1)
        self.clip = write_wav(self.tmp / "target.wav", self.target)
        episode = np.concatenate((melody(20.0, seed=2), self.target, melody(10.0, seed=3)))
        self.episode = write_wav(self.tmp / "episode.wav", episode)
        self.lib = Library.open(self.tmp / "lib")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_pair_from_clip_and_from_episode_match(self):
        a, _ = self.lib.add_pair("Spot A", self.clip, "target.wav")
        b, _ = self.lib.add_pair("Spot A", self.episode, "episode.wav", start_s=20.0, end_s=28.0)
        self.assertEqual((a.id, b.id), ("spot-a", "spot-a-2"))
        self.assertAlmostEqual(b.duration_s, 8.0, places=2)
        ta = HeadTemplate.load(self.lib.path(a.target))
        tb = HeadTemplate.load(self.lib.path(b.target))
        self.assertGreater(float(np.vdot(ta.head, tb.head)) / ta.width, 0.99)

        reloaded = Library.load(self.lib.root)
        self.assertEqual([p.id for p in reloaded.pairs], ["spot-a", "spot-a-2"])
        reloaded.validate()

    def test_episode_end_beyond_file_is_rejected(self):
        with self.assertRaises(LibraryError):
            self.lib.add_pair("x", self.episode, "episode.wav", start_s=30.0, end_s=90.0)
        self.assertEqual(self.lib.pairs, [])

    def test_replacement_trimmed_to_target(self):
        pair, _ = self.lib.add_pair("Spot", self.clip, "target.wav")
        long_clip = write_wav(self.tmp / "long.wav", melody(30.0, seed=5))
        self.lib.set_replacement(pair.id, long_clip, "long.wav")
        audio = decode(self.lib.path(pair.replacement), RATE, 2)
        self.assertEqual(len(audio), int(8.0 * RATE))
        self.assertLess(np.abs(audio[-100:]).max(), 0.01, "trimmed clip must fade out")
        self.assertIn("trimmed", pair.replacement_label)

        # Stored as 24-bit WAV at the library's rate, played straight from disk.
        from streamjanitor.audiofile import WavClip, load_clip, wav_layout
        channels, rate, width, _, _ = wav_layout(self.lib.path(pair.replacement))
        self.assertEqual((channels, rate, width), (2, self.lib.sample_rate, 3))
        clip = load_clip(self.lib.path(pair.replacement), self.lib.sample_rate, 2, fade_frames=0)
        self.assertIsInstance(clip, WavClip)
        self.assertEqual(len(clip), int(8.0 * self.lib.sample_rate))

        short_clip = write_wav(self.tmp / "short.wav", melody(2.0, seed=6))
        self.lib.set_replacement(pair.id, short_clip, "short.wav")
        self.assertEqual(len(decode(self.lib.path(pair.replacement), RATE, 2)), 2 * RATE)
        self.lib.remove_replacement(pair.id)
        self.assertIsNone(pair.replacement)

    def test_update_rebuilds_template_only_when_head_changes(self):
        pair, _ = self.lib.add_pair("Spot", self.clip, "target.wav")
        path = self.lib.path(pair.target)
        mtime = path.stat().st_mtime_ns
        self.lib.update(pair.id, name="New name", tau=0.7, enabled=False)
        self.assertEqual(path.stat().st_mtime_ns, mtime)
        self.lib.update(pair.id, head_s=2.0, head_offset=1.0)
        t = HeadTemplate.load(path)
        self.assertEqual((t.width, t.head_offset), (100, 50))
        with self.assertRaises(LibraryError):
            self.lib.update(pair.id, tau=1.5)
        with self.assertRaises(LibraryError):
            self.lib.update(pair.id, duration_s=3)

    def test_delete_removes_files(self):
        pair, _ = self.lib.add_pair("Spot", self.clip, "target.wav")
        self.lib.set_replacement(pair.id, self.clip, "target.wav")
        files = [self.lib.path(r) for r in (pair.target, pair.source, pair.replacement)]
        self.lib.delete(pair.id)
        self.assertFalse(any(f.exists() for f in files))
        self.assertEqual(Library.load(self.lib.root).pairs, [])

    def test_export_import_roundtrip(self):
        pair, _ = self.lib.add_pair("Spot", self.clip, "target.wav")
        self.lib.set_replacement(pair.id, self.clip, "target.wav")
        for include_sources in (True, False):
            with self.subTest(include_sources=include_sources):
                archive = self.tmp / "lib.zip"
                self.lib.export_zip(archive, include_sources)
                names = zipfile.ZipFile(archive).namelist()
                self.assertEqual(any(n.startswith("sources/") for n in names), include_sources)
                dest = self.tmp / "imported"
                imported = Library.import_archive(archive, dest)
                self.assertEqual([p.id for p in imported.pairs], ["spot"])
                imported.validate()

    def test_import_accepts_tar_with_top_folder_and_keeps_backup(self):
        self.lib.add_pair("Spot", self.clip, "target.wav")
        archive = self.tmp / "lib.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(self.lib.root, arcname="my-library")
        dest = self.tmp / "live"
        Library.open(dest)
        Library.import_archive(archive, dest)
        self.assertEqual(len(Library.load(dest).pairs), 1)
        self.assertTrue((self.tmp / "live.bak" / "library.toml").exists())

    def test_import_rejects_bad_archives(self):
        dest = self.tmp / "live"
        Library.open(dest)
        cases = {}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../evil.txt", "x")
        cases["path traversal"] = buf.getvalue()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("library.toml", 'version = 1\n[[pairs]]\nid = "a"\nname = "a"\nduration_s = 1.0\ntarget = "targets/a.npz"\n')
        cases["missing template"] = buf.getvalue()
        cases["not an archive"] = b"hello"
        for label, data in cases.items():
            with self.subTest(label):
                path = self.tmp / "bad.zip"
                path.write_bytes(data)
                with self.assertRaises(LibraryError):
                    Library.import_archive(path, dest)
                self.assertTrue((dest / "library.toml").exists(), "failed import must leave the library alone")
        self.assertFalse((self.tmp / "evil.txt").exists())

    def test_import_refuses_a_library_for_another_rate(self):
        self.lib.add_pair("Spot", self.clip, "target.wav")  # this library is 96 kHz (the default)
        archive = self.tmp / "lib.zip"
        self.lib.export_zip(archive)
        dest = self.tmp / "live"
        Library.open(dest, 48000)
        with self.assertRaisesRegex(LibraryError, "96000 Hz but this system runs at 48000"):
            Library.import_archive(archive, dest, sample_rate=48000)
        Library.import_archive(archive, dest, sample_rate=96000)

    def test_config_reads_library(self):
        pair, _ = self.lib.add_pair("Spot", self.clip, "target.wav")
        self.lib.update(pair.id, enabled=False)
        config = self.tmp / "config.toml"
        config.write_text('library = "lib"\n[audio]\ndelay_s = 4.0\n')
        cfg = load_config(config)
        self.assertEqual(len(cfg.targets), 1)
        self.assertFalse(cfg.targets[0].enabled)
        self.assertEqual(cfg.targets[0].name, "Spot")
        config.write_text('library = "lib"\n[audio]\nsample_rate = 48000\n')
        with self.assertRaisesRegex(ValueError, "96000"):
            load_config(config)


if __name__ == "__main__":
    unittest.main()

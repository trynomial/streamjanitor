import unittest

from streamjanitor.mopidy import caps_rate, loopback_output

ORIGINAL = "audioconvert ! audioresample ! audio/x-raw,rate=96000,channels=2,format=S24LE ! alsasink device=plughw:0,0"


class TestLoopbackOutput(unittest.TestCase):
    def test_keeps_the_users_chain_and_only_changes_the_sink(self):
        self.assertEqual(
            loopback_output(ORIGINAL, 96000),
            "audioconvert ! audioresample ! audio/x-raw,rate=96000,channels=2,format=S24LE ! "
            "alsasink device=plughw:Loopback,0 buffer-time=100000 latency-time=25000",
        )

    def test_is_idempotent(self):
        once = loopback_output(ORIGINAL, 96000)
        self.assertEqual(loopback_output(once, 96000), once)

    def test_rate_is_replaced_or_added(self):
        self.assertIn("rate=48000,channels=2,format=S24LE", loopback_output(ORIGINAL, 48000))
        out = loopback_output("audioconvert ! audio/x-raw,format=S32LE ! alsasink device=hw:1", 96000)
        self.assertEqual(out.split(" ! ")[:3], ["audioconvert", "audioresample", "audio/x-raw,format=S32LE,rate=96000"])
        out = loopback_output("autoaudiosink", 96000)  # no alsasink: default chain
        self.assertTrue(out.startswith("audioconvert ! audioresample ! audio/x-raw,rate=96000,channels=2,format=S24LE"))
        out = loopback_output("audioconvert ! alsasink", 96000)  # no caps: resampler + caps added
        self.assertEqual(out.split(" ! ")[:3], ["audioconvert", "audioresample", "audio/x-raw,rate=96000"])

    def test_other_sink_properties_survive(self):
        out = loopback_output("audioconvert ! alsasink sync=false device=hw:0 latency-time=10000", 96000)
        self.assertTrue(out.endswith("alsasink sync=false device=plughw:Loopback,0 buffer-time=100000 latency-time=25000"))

    def test_period_follows_alsa_latency(self):
        self.assertIn("buffer-time=200000 latency-time=50000", loopback_output(ORIGINAL, 96000, alsa_latency_ms=200))

    def test_caps_rate(self):
        self.assertEqual(caps_rate(ORIGINAL), 96000)
        self.assertEqual(caps_rate("audio/x-raw,rate=(int)44100 ! alsasink"), 44100)
        self.assertIsNone(caps_rate("audioconvert ! alsasink"))
        self.assertIsNone(caps_rate(None))


if __name__ == "__main__":
    unittest.main()

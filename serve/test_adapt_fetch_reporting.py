"""str-pk5: native admission policy -> real INFO parser -> HTTP /metrics (no GPU).

    python3 -m unittest serve.test_adapt_fetch_reporting -v

The CPU fixture calls the same policy as generate.cpp, with otherwise eligible
single-GPU predicates. It is not a native engine or a GPU coherence test.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, engine_args, serve
from serve.test_parallel import FAKE_BATCH

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("adapt_fetch_requested", "adapt_fetch_effective", "adapt_fetch_reason")


class AdaptFetchReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("CPU C++ compiler needed for native policy fixture")
        cls.build = tempfile.TemporaryDirectory(prefix="strata-adapt-fetch-test-")
        cls.policy = Path(cls.build.name) / "adapt_fetch_policy_test"
        subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "include"),
                        str(ROOT / "src/program/adapt_fetch_policy_test.cpp"), "-o", str(cls.policy)], check=True)
        subprocess.run([str(cls.policy)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = Path(self.tmp.name) / "fake_strata.py"
        real = server.subprocess.Popen
        self.spawn_patch = mock.patch.object(server.subprocess, "Popen",
            lambda cmd, **kw: real([sys.executable, str(self.script), *cmd[1:]], **kw))
        self.engine = self.httpd = None

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.engine:
            self.engine.unload()
        self.tmp.cleanup()

    def metadata(self, requested, batch):
        return subprocess.check_output([str(self.policy), "--info", str(requested), str(batch)], text=True).strip()

    def write_engine(self, metadata):
        self.script.write_text(FAKE_BATCH.replace('print("READY 4096 stop", flush=True)',
            f'print({metadata!r}, flush=True)\nprint("READY 4096 stop", flush=True)'), encoding="utf-8")

    def start(self, args, metadata):
        self.write_engine(metadata)
        with self.spawn_patch:
            self.engine = StrataEngine("strata", args)
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)

    def metrics(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.httpd.server_address[1]}/metrics", timeout=10) as r:
            return json.loads(r.read())

    def state(self):
        info = self.metrics()["engine"]
        return {key: info[key] for key in FIELDS if key in info}

    def test_native_driver_registers_policy_and_metrics_in_both_branches(self):
        source = (ROOT / "src/program/generate.cpp").read_text()
        # Bind the CPU policy regression to the actual driver, not an unused header.
        self.assertEqual(source.count("strata::program::adapt_fetch::evaluate(adapt_fetch(), {"), 2)
        self.assertEqual(source.count("INFO adapt_fetch_requested=%d adapt_fetch_effective=%d adapt_fetch_reason=%s"), 2)
        serving = source[source.index("const auto fetch_state ="):source.index("// #477 --expert-profile-save:")]
        self.assertIn(".batch = o.batch > 0", serving)
        self.assertIn("drive.d.fetch_admit = fetch_state.effective;", serving)
        self.assertIn("fetch_state.requested, drive.d.fetch_admit,", serving)
        self.assertLess(source.index("INFO adapt_fetch_requested="), source.index('std::printf("READY %lld stop'))

    def test_actual_production_config_batch_exclusion_and_solo_activity(self):
        cfg = json.loads((ROOT / "deploy/nibbler/config.json").read_text())
        self.assertEqual(cfg["parallel"], 2)
        self.assertEqual(cfg["env"]["STRATA_ADAPT_FETCH"], "2")
        with mock.patch.dict(os.environ, cfg["env"]):
            args = engine_args(cfg)
            batch = int(args[args.index("--batch") + 1])
            self.assertEqual(batch, 2)
            self.assertIn("--kv-unified", args)
            self.assertIn("--batch-mtp", args)
            # Not a fabricated Python eligibility policy: evaluate native shared policy.
            self.start(args, self.metadata(int(cfg["env"]["STRATA_ADAPT_FETCH"]), batch))
        expected = {"adapt_fetch_requested": 2, "adapt_fetch_effective": 0, "adapt_fetch_reason": "batch"}
        self.assertEqual(self.engine.batch, 2)
        self.assertEqual(self.state(), expected)
        self.assertIs(type(self.engine.info["adapt_fetch_requested"]), int)
        self.assertIs(type(self.engine.info["adapt_fetch_effective"]), int)
        self.assertIs(type(self.engine.info["adapt_fetch_reason"]), str)
        # One currently active request on GEN does not change engine configuration.
        run = self.svc.run([1, 2], False, False, 64, {}, threading.Event())
        try:
            next(run)
            self.assertTrue(self.engine.solo_active)
            self.assertEqual(self.state(), expected)
        finally:
            run.close()
        self.assertEqual(self.state(), expected)

    def test_modes_off_and_enabled_are_not_conflated(self):
        self.start([], self.metadata(1, 0))
        self.assertEqual(self.state(), {"adapt_fetch_requested": 1, "adapt_fetch_effective": 1,
                                       "adapt_fetch_reason": "none"})
        for requested, batch, reason in ((2, 0, "none"), (0, 2, "disabled")):
            self.write_engine(self.metadata(requested, batch))
            with self.spawn_patch:
                self.engine.restart()
            self.assertEqual(self.state(), {"adapt_fetch_requested": requested,
                                           "adapt_fetch_effective": requested,
                                           "adapt_fetch_reason": reason})

    def test_legacy_engine_does_not_guess_from_environment_or_batch(self):
        with mock.patch.dict(os.environ, {"STRATA_ADAPT_FETCH": "2"}):
            self.start(["--batch", "2"], "INFO kv_unified=1")
            self.assertEqual(self.state(), {})

    def test_restart_without_policy_info_does_not_retain_stale_state(self):
        self.start(["--batch", "2"], self.metadata(2, 2))
        self.assertEqual(self.state()["adapt_fetch_reason"], "batch")
        self.write_engine("INFO kv_unified=1")
        with self.spawn_patch:
            self.engine.restart()
        self.assertEqual(self.state(), {})
        self.assertEqual(self.engine.info["kv_unified"], 1)


if __name__ == "__main__":
    unittest.main()

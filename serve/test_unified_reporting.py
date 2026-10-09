"""/slots and /props reflect supported batch slots and the unified physical KV budget (no GPU).

    python -m unittest serve.test_unified_reporting -v
"""
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve
from serve.test_parallel import FAKE_BATCH


class UnifiedReporting(unittest.TestCase):
    def start(self, requested=0, fit=None, info=None, max_context=4096):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        # Extra startup INFO lines exercise the real parser, not just a hand-populated engine.info.
        metadata = "INFO " + " ".join(f"{k}={v}" for k, v in (info or {}).items())
        script.write_text(FAKE_BATCH.replace('print("READY 4096 stop", flush=True)',
                                            f'print({metadata!r}, flush=True)\nprint("READY {max_context} stop", flush=True)'),
                          encoding="utf-8")
        args = ["--batch", str(requested)] if requested else []
        if fit is not None:
            args += ["--fit", str(fit)]
        real = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", args)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def assert_slots(self, processing, prompt_tokens=None):
        if prompt_tokens is None:
            prompt_tokens = [0] * len(processing)
        self.assertEqual(len(prompt_tokens), len(processing))
        self.assertEqual(self.get("/slots"),
                         [{"id": b, "n_ctx": self.engine.max_context, "is_processing": active,
                           "n_prompt_tokens": prompt_tokens[b]}
                          for b, active in enumerate(processing)])
        count = len(processing)
        self.assertEqual(self.get("/props")["total_slots"], count)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], count)

    def test_loaded_solo_engine(self):
        self.start()
        self.assertEqual(self.engine.batch, 0)
        self.assert_slots([False])
        with self.svc.status_lock:
            self.svc.status.update(busy=True, queued=3)
        self.assert_slots([True])
        with self.svc.status_lock:
            self.svc.status.update(busy=False)
        self.assert_slots([False])   # queued requests alone are not processing
        props = self.get("/props")
        for key in ("kv_unified", "kv_capacity_cells", "kv_resident", "kv_resident_capacity_cells"):
            self.assertNotIn(key, props)

    def test_fake_engine_without_info_has_no_capacity_metadata(self):
        self.start()
        fake = MockEngine(ByteTokenizer(), "ok", max_context=262144)
        with mock.patch.object(self.svc, "engine", fake):
            props = self.get("/props")
            self.assertEqual(props["total_slots"], 1)
            self.assertEqual(props["default_generation_settings"]["n_ctx"], 262144)
            self.assertEqual(self.get("/slots"), [{"id": 0, "n_ctx": 262144, "is_processing": False,
                                                   "n_prompt_tokens": 0}])
            for key in ("kv_unified", "kv_capacity_cells", "kv_resident", "kv_resident_capacity_cells"):
                self.assertNotIn(key, props)

    def test_supported_multi_slots_and_per_slot_activity(self):
        self.start(requested=8, fit=3)
        self.assertEqual(self.engine.batch, 3)
        self.assert_slots([False, False, False])
        # Cached conversations and reserved slots are not the same as processing requests.
        self.engine.slot_held[0] = [1, 2, 3]
        self.engine.slot_busy[:] = [True, True, True]
        self.engine.slot_live[:] = [None, {"state": "reading"}, {"state": "decoding"}]
        with self.svc.status_lock:
            self.svc.status.update(busy=True, queued=4)
        self.assert_slots([False, True, True], prompt_tokens=[3, 0, 0])
        self.engine.slot_live[:] = [None] * 3
        # Service busy can also mean waiting for admission: don't invent a solo request.
        self.assert_slots([False, False, False], prompt_tokens=[3, 0, 0])
        with self.svc.status_lock:
            self.svc.status.update(busy=False, queued=0)
        self.engine.slot_busy[:] = [False] * 3

    def test_requested_batch_unsupported_falls_back_to_one(self):
        self.start(requested=4, fit=0)
        self.assertEqual(self.engine.batch, 0)
        self.assert_slots([False])
        with self.svc.status_lock:
            self.svc.status["busy"] = True
        self.assert_slots([True])

    def test_unloaded_has_no_slots_and_does_not_reload(self):
        self.start(requested=4, fit=2)
        self.engine.unload()
        with mock.patch.object(self.engine, "restart", side_effect=AssertionError("reporting must not reload")):
            self.assertEqual(self.get("/slots"), [])
            props = self.get("/props")
            self.assertTrue(props["is_sleeping"])
            self.assertEqual(props["total_slots"], 2)

    def test_unified_context_is_logical_not_divided_or_multiplied(self):
        self.start(requested=4, fit=3, info={"kv_unified": 1, "kv_capacity_cells": 4096})
        self.assert_slots([False, False, False])
        props = self.get("/props")
        self.assertIs(props["kv_unified"], True)
        self.assertEqual(props["kv_capacity_cells"], 4096)
        self.assertEqual(props["default_generation_settings"]["n_ctx"], 4096)
        self.assertEqual(self.engine.info["kv_unified"], 1)
        self.assertEqual(self.engine.info["kv_capacity_cells"], 4096)

    def test_unified_streaming_keeps_logical_backing_and_gpu_capacity_distinct(self):
        info = {"kv_unified": 1, "kv_capacity_cells": 262144, "kv_resident": 32768,
                "kv_resident_capacity_cells": 32768}
        self.start(requested=4, fit=2, info=info, max_context=262144)
        self.assert_slots([False, False])
        props = self.get("/props")
        self.assertIs(props["kv_unified"], True)
        for key, value in info.items():
            self.assertEqual(self.engine.info[key], value)
            self.assertEqual(props[key], value)
        self.assertEqual(props["default_generation_settings"]["n_ctx"], 262144)
        self.assertEqual([slot["n_ctx"] for slot in self.get("/slots")], [262144, 262144])

    def test_incremental_capability_reports_headroom_not_a_logical_output_cap(self):
        self.start(requested=2, max_context=262144,
                   info={"kv_unified": 1, "kv_capacity_cells": 262144, "kv_resident": 32768,
                         "kv_incremental": 1, "kv_reserve_ahead": 256})
        props = self.get("/props")
        self.assertIs(props["kv_incremental"], True)
        self.assertEqual(props["kv_reserve_ahead"], 256)
        self.assertEqual(props["default_generation_settings"]["n_ctx"], 262144)
        self.assertEqual(props["default_generation_settings"]["params"]["n_predict"], -1)
        self.assertEqual([slot["n_ctx"] for slot in self.get("/slots")], [262144, 262144])
        metrics = self.get("/metrics")
        self.assertEqual(metrics["engine"]["kv_incremental"], 1)
        self.assertEqual(metrics["engine"]["kv_reserve_ahead"], 256)
        self.assertEqual(metrics["live"]["pressure_waiting"], [])

    def test_unified_fully_resident_reports_actual_rounded_capacity(self):
        self.start(requested=2, max_context=4097,
                   info={"kv_unified": 1, "kv_capacity_cells": 4100, "kv_resident": 0,
                         "kv_resident_capacity_cells": 4100})
        self.assert_slots([False, False])
        props = self.get("/props")
        self.assertEqual(props["default_generation_settings"]["n_ctx"], 4097)
        self.assertEqual(props["kv_capacity_cells"], 4100)
        self.assertEqual(props["kv_resident"], 0)
        self.assertEqual(props["kv_resident_capacity_cells"], 4100)

    def test_non_unified_metadata_is_not_treated_as_shared(self):
        self.start(requested=2, max_context=262144,
                   info={"kv_unified": 0, "kv_capacity_cells": 0, "kv_resident": 32768})
        self.assert_slots([False, False])
        props = self.get("/props")
        self.assertIs(props["kv_unified"], False)
        self.assertEqual(props["kv_capacity_cells"], 0)
        self.assertEqual(props["kv_resident"], 32768)
        self.assertNotIn("kv_resident_capacity_cells", props)  # don't guess it from residency or slots

    def test_legacy_fully_resident_zero_is_exposed_without_other_metadata(self):
        self.start(info={"kv_resident": 0})
        self.assert_slots([False])
        props = self.get("/props")
        self.assertEqual(props["kv_resident"], 0)
        for key in ("kv_unified", "kv_capacity_cells", "kv_resident_capacity_cells"):
            self.assertNotIn(key, props)

    def test_unified_without_capacity_does_not_guess_physical_budget(self):
        self.start(requested=2, info={"kv_unified": 1})
        props = self.get("/props")
        self.assertIs(props["kv_unified"], True)
        for key in ("kv_capacity_cells", "kv_resident", "kv_resident_capacity_cells"):
            self.assertNotIn(key, props)

    def test_batch_capable_engine_actually_running_solo_and_cancelled(self):
        self.start(requested=3)
        # Suspend the consumer while GEN is active. slot_live stays empty on this fast solo path.
        run = self.svc.run([1, 2], False, False, 64, {}, threading.Event())
        try:
            next(run)
            self.assertTrue(self.engine.solo_active)
            self.assertEqual(self.engine.slot_live, [None] * 3)
            self.assert_slots([True, False, False], prompt_tokens=[2, 0, 0])
        finally:
            run.close()   # STOP and drain; the solo-active flag must not stick after cancellation
        self.assertFalse(self.engine.solo_active)
        self.assert_slots([False, False, False])
        # And completion on the same engine clears the flag too.
        list(self.svc.run([1, 2], False, False, 64, {}, threading.Event()))
        self.assertFalse(self.engine.solo_active)
        self.assert_slots([False, False, False])


if __name__ == "__main__":
    unittest.main()

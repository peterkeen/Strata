#!/usr/bin/env python3
"""Offline Q4_K_XL performance plan and log report helper.

This tool is deliberately a plan/report generator.  ``--dry-run`` reads only the
named config and local files needed for identity fingerprints.  It never starts
Strata, opens sockets, calls HTTP endpoints, unloads services, or changes the
machine.  Live execution requires a separate, approved harness/window; this file
only makes those later runs reproducible and parseable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

# A first dry run must not leave __pycache__ behind.
sys.dont_write_bytecode = True

GIB = 1024 ** 3
MIB = 1024 ** 2
DEFAULT_REPEATS = 3
RESIDUAL_PROMPT_TOKENS = (31, 64, 66, 256, 512, 1024, 2048)
PREFILL_TOKENS = (8192, 32768)
DECODE_TOKENS = 256


class Stop(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except OSError as e:
        raise Stop(f"cannot read config: {e}") from e
    except json.JSONDecodeError as e:
        raise Stop(f"config is not JSON: {e}") from e


def flag_value(args: list[Any], name: str) -> str | None:
    values = [str(args[i + 1]) for i, v in enumerate(args[:-1]) if v == name]
    return values[-1] if values else None


def has_flag(args: list[Any], name: str) -> bool:
    return any(v == name for v in args)


def executable_path(cfg: dict[str, Any]) -> str | None:
    exe = cfg.get("exe") or cfg.get("executable")
    return str(exe) if exe else None


def file_identity(path_text: str | None) -> dict[str, Any]:
    if not path_text:
        return {"path": None, "exists": False, "sha256": None, "bytes": None}
    path = Path(path_text)
    out: dict[str, Any] = {"path": str(path), "exists": path.exists(), "sha256": None, "bytes": None}
    if path.is_file():
        st = path.stat()
        out.update(bytes=st.st_size, sha256=sha256_file(path))
    return out


def config_fingerprints(config_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    args = [str(x) for x in cfg.get("args", [])]
    semantic = {
        "model_name": cfg.get("model_name"),
        "parallel": cfg.get("parallel"),
        "exe": executable_path(cfg),
        "cwd": cfg.get("cwd"),
        "log": cfg.get("log"),
        "tokenizer": cfg.get("tokenizer"),
        "pack": flag_value(args, "--pack"),
        "native": flag_value(args, "--native"),
        "kv": flag_value(args, "--kv"),
        "kv_resident": flag_value(args, "--kv-resident"),
        "max_context": flag_value(args, "--max-context"),
        "prefill": flag_value(args, "--prefill"),
        "spec": flag_value(args, "--spec"),
        "spec_min_p": flag_value(args, "--spec-min-p"),
        "batch_mtp": has_flag(args, "--batch-mtp"),
        "kv_unified": has_flag(args, "--kv-unified"),
        "resident_budget_gib": flag_value(args, "--resident-budget-gib"),
        "resident_experts": has_flag(args, "--resident-experts"),
        "expert_profile": flag_value(args, "--expert-profile"),
        "conversation_cache_mib": flag_value(args, "--conversation-cache-mib"),
        "conversation_cache_slots": flag_value(args, "--conversation-cache-slots"),
        "env": {k: cfg.get("env", {}).get(k) for k in sorted(cfg.get("env", {}))
                if k.startswith("STRATA_")},
    }
    semantic_json = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
    return {
        "config": file_identity(str(config_path)),
        "executable": file_identity(executable_path(cfg)),
        "semantic_config": semantic,
        "semantic_sha256": sha256_text(semantic_json),
        "identity_note": "Fingerprint only the named executable and config semantics; model payloads are identified by paths/manifests, not rehashed here.",
    }


def validate_q4_config(cfg: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    args = [str(x) for x in cfg.get("args", [])]
    if cfg.get("parallel") != 2:
        warnings.append("expected parallel=2 for nibbler Q4 acceptance matrix")
    for flag, want in (("--max-context", "262144"), ("--kv", "int8"), ("--kv-resident", "32768")):
        got = flag_value(args, flag)
        if got != want:
            warnings.append(f"expected {flag}={want}, saw {got!r}")
    for flag in ("--kv-unified",):
        if not has_flag(args, flag):
            warnings.append(f"expected {flag} for unified-KV semantics")
    pack = flag_value(args, "--pack") or ""
    if pack and not re.search(r"q4|Q4|UD-Q4|qwen", pack):
        warnings.append("pack path does not look like the sanitized Q4_K_XL fixture; verify before live use")
    return warnings


def workloads() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in RESIDUAL_PROMPT_TOKENS:
        rows.append({
            "id": f"residual-{n}", "kind": "residual_prompt", "new_prompt_tokens": n,
            "decode_tokens": DECODE_TOKENS, "slots": ["solo", "overlap-2"],
            "reuse": "bounded same-conversation prefix; prompt suffix changes only",
        })
    for n in PREFILL_TOKENS:
        rows.append({
            "id": f"prefill-{n}", "kind": "fresh_prefill", "new_prompt_tokens": n,
            "decode_tokens": DECODE_TOKENS, "slots": ["solo", "overlap-2"],
            "reuse": "fresh, then exact warm repeat to bound interleaved conversation reuse",
        })
    rows.append({
        "id": "decode-256", "kind": "decode", "new_prompt_tokens": 256,
        "decode_tokens": DECODE_TOKENS, "slots": ["solo", "overlap-2"],
        "reuse": "bounded interleaved conversation chain of at most four turns; no unbounded history growth",
    })
    return rows


def arms() -> list[dict[str, Any]]:
    return [
        {"id": "baseline", "changes": [], "purpose": "same-day control; no feature change"},
        {"id": "short-read-crossover", "changes": [{"setting": "--short-read", "value": "candidate"}],
         "purpose": "one-change candidate for residual prompt crossover"},
        {"id": "quiescent-cpu-prefill-share", "changes": [{"setting": "prefill CPU-share lease", "value": "candidate opt-in"}],
         "purpose": "one-change candidate for small/medium prefills"},
        {"id": "slot-aware-pcie-balance", "changes": [{"setting": "--pcie-frac / slot-aware balancer", "value": "candidate opt-in"}],
         "purpose": "one-change candidate for decode split"},
        {"id": "batch-mtp-q4", "changes": [{"setting": "--batch-mtp", "value": "candidate enabled"}],
         "purpose": "one-change candidate for overlapping-slot decode"},
        {"id": "demand-parallel-resources", "changes": [{"setting": "parallel slot/draft allocation", "value": "candidate opt-in"}],
         "purpose": "one-change candidate for slot memory pressure"},
    ]


def interleaved_order(repeats: int = DEFAULT_REPEATS) -> list[dict[str, Any]]:
    if not 1 <= repeats <= 10:
        raise Stop("repeats must be 1..10")
    out: list[dict[str, Any]] = []
    candidate_ids = [a["id"] for a in arms() if a["id"] != "baseline"]
    for rep in range(repeats):
        rotated = candidate_ids[rep % len(candidate_ids):] + candidate_ids[:rep % len(candidate_ids)]
        for workload in workloads():
            for candidate in rotated:
                if rep % 2 == 0:
                    pair = ["baseline", candidate]
                else:
                    pair = [candidate, "baseline"]
                for arm in pair:
                    out.append({"repeat": rep + 1, "workload": workload["id"], "arm": arm})
    return out


def stop_gates() -> dict[str, Any]:
    return {
        "memory": {
            "MemAvailable_floor_bytes": 10 * GIB,
            "SwapFree_floor_bytes": 1 * GIB,
            "record": ["MemAvailable", "SwapFree", "Cached", "AnonPages", "MemFree", "VmHWM", "VmRSS"],
        },
        "gpu_vram": {
            "record_high_water": True,
            "stop_if_free_vram_below_mib": "freshly approved run-specific reserve; do not infer from dry run",
        },
        "engine": {
            "stop_on": ["ERR", "out of memory", "cudaMalloc failure", "HIP error", "segmentation fault", "bad_alloc"],
            "engine_silence_s": "use serve/server.py watchdog policy for an authorized live run",
        },
        "service_safety": {
            "dry_run": "no subprocess, no socket, no HTTP, no service control, no GPU/model request",
            "live_prereq": "separate isolated window and str-inn.5 safety revalidation",
        },
        "io_accounting": {
            "physical": "/proc/self/io read_bytes and major faults from engine log",
            "logical": "expert blob MB/file tier counters from DONE and expert-tier lines",
            "rule": "never treat logical blob bytes as physical SSD reads",
        },
    }


def dry_run_report(config_path: Path, repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    cfg = load_config(config_path)
    return {
        "schema": "strata.q4_perf_plan.v1",
        "mode": "dry-run-offline-plan",
        "safety": "CPU-only report generation; no engine launch, network, HTTP, service control, GPU access, or production request.",
        "identities": config_fingerprints(config_path, cfg),
        "config_warnings": validate_q4_config(cfg),
        "workloads": workloads(),
        "arms": arms(),
        "interleaved_order": interleaved_order(repeats),
        "stop_gates": stop_gates(),
        "required_measurements": [
            "prompt_read_ms/tokens separately from queue/wall time",
            "loan/refill and adaptive wait separately",
            "CPU expert pool, GPU/PCIe wait and routed shares separately; overlapping times are not summed",
            "logical blob bytes and physical read_bytes separately",
            "drafts offered/accepted/rejected, routed VRAM/PCIe/CPU shares, memory high-water marks",
            "solo and overlapping slots with bounded conversation reuse",
        ],
        "observed_historical_traffic": {
            "label": "historical uncontrolled production observations from docs/Q4_NIBBLER_PERFORMANCE_AUDIT.md",
            "not_matched_ab": True,
            "examples": [
                {"new_prompt_tokens": 31, "reused_tokens": 20135, "read_ms": 790, "read_tok_s": 39.2},
                {"new_prompt_tokens": 66, "reused_tokens": 9159, "read_ms": 1486, "read_tok_s": 44.4},
                {"generated_tokens": 10179, "decode_ms": 351196, "decode_tok_s": 29.0},
            ],
        },
        "matched_ab_result": {
            "status": "not_run",
            "speedup_claim": None,
            "note": "This plan never claims speedup. Report medians/tails only after repeated interleaved baseline/candidate trials.",
        },
    }


class LogParser:
    REQUEST = re.compile(r"strata serve: prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ (?P<read_text>\d+(?: of \d+)?) read in (?P<prompt_ms>[\d.]+) ms \((?P<prompt_tps>[\d.]+) tok/s\), (?P<gen>\d+) generated in (?P<decode_ms>[\d.]+) ms \((?P<decode_tps>[\d.]+) tok/s\), drafts accepted (?P<draft_acc>\d+) of (?P<draft_off>\d+), (?P<checkpoints>\d+) checkpoints(?P<cancelled> \(cancelled\))?")
    DONE = re.compile(r"^DONE (?P<generated>\d+) (?P<prompt>\d+) (?P<prompt_ms>[\d.]+) (?P<decode_ms>[\d.]+) (?P<finish>\S+)(?: (?P<draft_acc>\d+) (?P<draft_off>\d+) (?P<reused>\d+)(?: (?P<hits>\d+) (?P<lookups>\d+)(?: (?P<ram_blobs>\d+) (?P<file_blobs>\d+) (?P<file_mb>[\d.]+)(?: (?P<prompt_read>\d+)(?: (?P<offloaded>\d+))?)?)?)?)?")
    BATCH = re.compile(r"strata batch: (?P<windows>\d+) windows, avg (?P<rows>[\d.]+) rows, (?P<ms_window>[\d.]+) ms/window = run (?P<run>[\d.]+) \(CUDA0 GPU-reach wait (?P<gpu_wait>[\d.]+) \+ CPU experts (?P<cpu_experts>[\d.]+)\) \+ commit (?P<commit>[\d.]+) \+ emit (?P<emit>[\d.]+) \+ adapt wait (?P<adapt>[\d.]+);.*?routed entries per window: VRAM (?P<vram>[\d.]+) \((?P<vram_pct>[\d.]+)%\), PCIe (?P<pcie>[\d.]+) \((?P<pcie_pct>[\d.]+)%\), CPU (?P<cpu>[\d.]+) \((?P<cpu_pct>[\d.]+)%\)")
    STAGE = re.compile(r"strata serve: stage (?P<stage>\d+): (?P<windows>\d+) windows; per window: wait for the GPU (?P<wait>[\d.]+) ms, pool \+ plan (?P<pool>[\d.]+) ms, host staging (?P<host>[\d.]+) ms, commit (?P<commit>[\d.]+) ms")
    PREFILL = re.compile(r"strata prefill timing: (?P<tokens>\d+) tokens, GPU timeline (?P<gpu>[\d.]+) ms, wall (?P<wall>[\d.]+) ms, host staging (?P<host>[\d.]+) ms:(?P<parts>.*)")
    PREFILL_HOST = re.compile(r"strata prefill timing: host: chunk setup .*? (?P<setup>[\d.]+) ms, waiting for each chunk (?P<waiting>[\d.]+) ms, after each chunk .*? (?P<after>[\d.]+) ms, PLE (?P<ple>[\d.]+) ms")
    EXPERT_TIERS = re.compile(r"strata serve: expert tiers: GPU (?P<gpu_hits>\d+) hits this request; since the start RAM (?P<ram>\d+) blobs, files (?P<files>\d+) blobs (?P<mb>[\d.]+) MB read")
    FILE_IO = re.compile(r"strata serve: file tier I/O this request: the OS read (?P<physical_mb>[\d.]+) MB from storage \((?P<faults>\d+) major faults\) for (?P<logical_mb>[\d.]+) MB of expert reads(?:; page cache held (?P<cached_mb>[\d.]+) MB at the read, not (?P<uncached_mb>[\d.]+) MB)?")
    RESIDENT = re.compile(r"strata serve: resident RAM: (?P<gib>[\d.]+) GiB of experts in RAM, (?P<exchanged>\d+) exchanged with the VRAM tier, (?P<files>\d+) blob reads from the file")
    BATCH_MTP = re.compile(r"strata batch_mtp_stats slot=(?P<slot>\d+) windows=(?P<windows>\d+) offered=(?P<offered>\d+) accepted=(?P<accepted>\d+) rejected=(?P<rejected>\d+) discarded=(?P<discarded>\d+) fallback_attempts=(?P<fallback_attempts>\d+) fallback_incoherent=(?P<fallback_incoherent>\d+) fallback_not_ready=(?P<fallback_not_ready>\d+) fallback_limits=(?P<fallback_limits>\d+) fallback_capacity=(?P<fallback_capacity>\d+) fallback_reserve=(?P<fallback_reserve>\d+)")
    FRONTEND_REQUEST = re.compile(r"\[strata\] request prompt (?P<prompt>\d+) cached (?P<cached>\d+) output (?P<output>\d+) prompt_read (?P<prompt_ms>[\d.]+) ms total (?P<total_ms>[\d.]+) ms prefill (?P<prefill_tps>[\d.]+) tok/s decode (?P<decode_tps>[\d.]+) tok/s")
    MEMORY_JSON_KEYS = {"MemAvailable", "SwapFree", "VmHWM", "VmRSS"}

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.done: list[dict[str, Any]] = []
        self.batch_windows: list[dict[str, Any]] = []
        self.stage_windows: list[dict[str, Any]] = []
        self.prefill: list[dict[str, Any]] = []
        self.file_io: list[dict[str, Any]] = []
        self.expert_tiers: list[dict[str, Any]] = []
        self.resident_ram: list[dict[str, Any]] = []
        self.batch_mtp: list[dict[str, Any]] = []
        self.frontend_requests: list[dict[str, Any]] = []
        self.memory_high_water: dict[str, Any] = {}
        self.warnings: list[str] = []
        self._prev_expert_tier: dict[str, float] | None = None

    @staticmethod
    def _numbers(match: re.Match[str]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in match.groupdict().items():
            if value is None:
                continue
            if re.fullmatch(r"\d+", value):
                row[key] = int(value)
            elif re.fullmatch(r"\d+\.\d+", value):
                row[key] = float(value)
            else:
                row[key] = value
        return row

    def parse_text(self, text: str) -> dict[str, Any]:
        if text and not text.endswith("\n"):
            self.warnings.append("log ended with a truncated partial line; ignored partial tail")
        for raw in text.splitlines():
            self.parse_line(raw)
        if not self.requests and not self.done:
            self.warnings.append("no complete request/DONE lines found")
        return self.report()

    def parse_line(self, line: str) -> None:
        if m := self.REQUEST.search(line):
            row = self._numbers(m)
            read_text = str(row.pop("read_text"))
            row["prompt_read_tokens"] = int(read_text.split()[0])
            row["cancelled"] = bool(row.pop("cancelled", ""))
            self.requests.append(row)
            return
        if m := self.DONE.search(line):
            row = self._numbers(m)
            if "hits" in row and "lookups" in row:
                routed = row.get("lookups", 0) + row.get("offloaded", 0)
                row["vram_share"] = row["hits"] / row["lookups"] if row["lookups"] else None
                row["pcie_share"] = row["offloaded"] / routed if routed else None
            self.done.append(row)
            return
        if m := self.BATCH.search(line):
            row = self._numbers(m)
            row["components_overlap"] = True
            row["do_not_sum_as_disjoint"] = ["gpu_wait_ms", "cpu_experts_ms", "adapt_ms", "run_ms"]
            self.batch_windows.append(row)
            return
        if m := self.STAGE.search(line):
            self.stage_windows.append(self._numbers(m)); return
        if m := self.PREFILL.search(line):
            row = self._numbers(m)
            row["parts"] = self._parse_prefill_parts(str(row.get("parts", "")))
            self.prefill.append(row); return
        if m := self.PREFILL_HOST.search(line):
            row = self._numbers(m)
            if self.prefill:
                self.prefill[-1]["host_breakdown"] = row
            else:
                self.prefill.append({"host_breakdown": row})
            return
        if m := self.EXPERT_TIERS.search(line):
            row = self._numbers(m)
            cumulative = {"ram_blobs": row["ram"], "file_blobs": row["files"], "logical_mb": row["mb"]}
            if self._prev_expert_tier is None:
                delta = dict(cumulative)
            else:
                delta = {k: cumulative[k] - self._prev_expert_tier[k] for k in cumulative}
            row["cumulative"] = cumulative
            row["delta_since_previous_line"] = delta
            row["logical_blob_bytes_only"] = True
            self._prev_expert_tier = dict(cumulative)
            self.expert_tiers.append(row); return
        if m := self.FILE_IO.search(line):
            row = self._numbers(m)
            row["physical_read_bytes"] = int(row["physical_mb"] * 1e6)
            row["logical_blob_bytes"] = int(row["logical_mb"] * 1e6)
            self.file_io.append(row); return
        if m := self.RESIDENT.search(line):
            self.resident_ram.append(self._numbers(m)); return
        if m := self.BATCH_MTP.search(line):
            self.batch_mtp.append(self._numbers(m)); return
        if m := self.FRONTEND_REQUEST.search(line):
            self.frontend_requests.append(self._numbers(m)); return
        try:
            obj = json.loads(line)
        except ValueError:
            return
        if isinstance(obj, dict) and self.MEMORY_JSON_KEYS.intersection(obj):
            for key in ("VmHWM", "VmRSS", "MemAvailable", "SwapFree"):
                if key in obj:
                    self.memory_high_water[key] = max(self.memory_high_water.get(key, 0), obj[key])

    @staticmethod
    def _parse_prefill_parts(parts: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, ms, pct in re.findall(r"([A-Za-z0-9_/-]+) ([\d.]+) \(([\d.]+)%\)", parts):
            out[name] = float(ms)
            out[name + "_pct"] = float(pct)
        return out

    def report(self) -> dict[str, Any]:
        return {
            "schema": "strata.q4_perf_report.v1",
            "report_kind": "parsed_log_or_fixture",
            "matched_ab": False,
            "speedup_claim": None,
            "note": "Parsed timings are stage costs only. CPU/GPU/PCIe/adaptive times may overlap and are not added as disjoint speedup evidence.",
            "requests": self.requests,
            "done": self.done,
            "batch_windows": self.batch_windows,
            "stage_windows": self.stage_windows,
            "prefill": self.prefill,
            "file_io": self.file_io,
            "expert_tiers": self.expert_tiers,
            "resident_ram": self.resident_ram,
            "batch_mtp": self.batch_mtp,
            "frontend_requests": self.frontend_requests,
            "memory_high_water": self.memory_high_water,
            "warnings": self.warnings,
        }


def parse_log(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        raise Stop(f"cannot read log: {e}") from e
    return LogParser().parse_text(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="sanitized Q4 config JSON to fingerprint")
    ap.add_argument("--dry-run", action="store_true", help="emit offline plan; the default is also non-executing")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--parse-log", type=Path, help="optional engine/server log fixture to parse into a stage-cost report")
    args = ap.parse_args(argv)
    if args.parse_log:
        out = {"plan": dry_run_report(args.config, args.repeats), "parsed_report": parse_log(args.parse_log)}
    else:
        out = dry_run_report(args.config, args.repeats)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Stop as e:
        print(f"q4_perf_plan: {e}", file=sys.stderr)
        raise SystemExit(2)

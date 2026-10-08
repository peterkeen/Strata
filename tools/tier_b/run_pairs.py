"""tools/tier_b/run_pairs.py - focused same-binary first-wave model A/B pairs (sta-osh.9).

Two INDEPENDENT default-versus-opt-out families, nothing else:

    normrope : default (fused native RMSNorm+RoPE)  vs  STRATA_NO_NORM_ROPE=1 (split)
    sub16    : default (grouped GU/down split_sub16) vs  STRATA_NO_SUB16_GU=1  (reference)

Each pair runs the SAME Tier-B executable with identical inputs and common
settings; the only difference between the A (default) and B (opt-out) arm is
exactly one family override.  A combined "all-off" arm is refused by
construction (assert_single_family_override).

Two strictly separated run modes:

  * DIAGNOSTIC (--mode diagnostic): STRATA_TIER_B_TRACE=1 plus the bounded
    --tier-b-export DIR from sta-osh.5.  These are CORRECTNESS arms only;
    their wall-clock/decode numbers are NEVER used as timing samples.
  * TIMING (--mode timing): all diagnostics OFF; interleaved, counterbalanced
    AB/BA pairs whose decode/prefill metrics feed the paired statistics.

This module launches engines ONLY through the hardened Tier-A runner
(tier_a.engine_cli.run_arm): process-group cancellation, tokens-file temp
handling and stdout parsing are inherited, never re-implemented here.  No
shell, no legacy orphan kills.  --dry-run launches ZERO engines.

OFFLINE SCOPE
─────────────
This harness and its parsers/validators run with no GPU, no network and no
model.  Live execution (not --dry-run) is the PRIMARY's host gate (sta-osh.11)
after sta-osh.10; this file asserts nothing about the host.  The tier_a fast
suite does not discover this directory; run it explicitly:

    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONPATH=/home/pete/.local/share/uv/tools/nanobot-ai/lib/python3.11/site-packages \\
    python3 -m unittest discover -s tools/tier_b -p 'test_*.py' -q
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

# Launch and parsing are inherited from the hardened Tier-A runner (sta-d76 /
# sta-0a7).  run_arm is bound at module scope so offline tests can monkeypatch
# tools.tier_b.run_pairs.run_arm and prove the dry-run path never launches.
from tier_a.engine_cli import ArmResult, run_arm  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# PREDECLARED CONTRACTS AND CONSTANTS  (fitted-after-the-fact values are refused)
# ─────────────────────────────────────────────────────────────────────────────

# Common STRATA additions applied to EVERY model arm (plan: "All model arms").
COMMON_STRATA: dict[str, str] = {"STRATA_PF_FUSED": "1", "STRATA_IQ_MT_MIN": "1"}

# BOTH first-wave families are identity arms (plan "Evidence and numerical
# contracts": exact expectations for these identity arms; primitive tolerances
# are never a blanket full-model logit tolerance).
#   normrope: the fused kernel is bit-identical to native_qsa_rms_norm_weighted +
#             native_rope_apply (source contract, rope_parity check 6, host PASS
#             192 cases bitwise).  The split rope's 32- vs 128-thread launch is
#             per-pair independent arithmetic, so the model opt-out must match
#             bit-for-bit.  A difference is a defect to diagnose, not a reason to
#             loosen the contract after the fact.
#   sub16:    layout-only change over identical quantized bytes and reduction
#             arithmetic (iq_multi_parity host PASS bitwise).
NORMROPE_REL_TOL = 0.0  # bitwise
SUB16_REL_TOL = 0.0     # bitwise

# Diagnostic trace identifiers (src/kernels/cuda/native_rope.cu, iq_kernels.cu).
# A positive record must be exactly these; unknown/skipped/unsupported/zero/
# truncated records can never count as positive proof.
_ROPE_PLAIN_PATHS = ("rope_inplace32", "rope128")
_SUB16_PATH = "split_sub16"
# A down/GU reference path for the B (opt-out) arm at the studied geometry.
_SUB16_REF_PATHS = ("split", "reference", "reference_old")
# Records that are never positive evidence for either arm.
_NONPOSITIVE_PATHS = ("skipped", "unsupported", "zero", "truncated", "format_error")

# Studied geometry (plan: n_embd=2560 GU rows, n_ff=640 down rows).
SUB16_N_EMBD = 2560
SUB16_N_FF = 640


@dataclass(frozen=True)
class Family:
    name: str
    override_env: str          # the single STRATA opt-out applied to arm B
    contract: str              # "bitwise" or "tolerance"
    rel_tol: float             # 0.0 when bitwise

    def validate(self) -> None:
        if self.contract not in ("bitwise", "tolerance"):
            raise ValueError(f"family {self.name}: bad contract {self.contract!r}")
        if self.contract == "bitwise" and self.rel_tol != 0.0:
            raise ValueError(f"family {self.name}: bitwise contract must have rel_tol 0.0")
        if self.contract == "tolerance" and not (self.rel_tol > 0.0):
            raise ValueError(f"family {self.name}: tolerance contract needs positive rel_tol")


FAMILIES: dict[str, Family] = {
    "normrope": Family("normrope", "STRATA_NO_NORM_ROPE", "bitwise",  NORMROPE_REL_TOL),
    "sub16":    Family("sub16",    "STRATA_NO_SUB16_GU",  "bitwise",  SUB16_REL_TOL),
}

# Every family opt-out var.  Used to refuse a combined all-off arm.
FAMILY_OVERRIDE_VARS: tuple[str, ...] = tuple(f.override_env for f in FAMILIES.values())

# Diagnostic-only controls (plan "Proof runs only").  Forbidden in clean timing.
DIAGNOSTIC_VARS: tuple[str, ...] = ("STRATA_TIER_B_TRACE", "STRATA_TIER_B_EXPORT")
DIAGNOSTIC_FLAGS: tuple[str, ...] = ("--tier-b-export", "--dump-logits")

WORKLOADS: tuple[str, ...] = ("short_code", "long_doc")

# Predeclared live prompt token-id counts (from the Tier-A host records).  A live
# run refuses to launch if the tokenizer produces a different count, so a stale
# corpus or tokenizer can never silently change the compared inputs.
EXPECTED_PROMPT_IDS: dict[str, int] = {"short_code": 42, "long_doc": 8887}
# sha256[:16] of the space-joined ids (tier_a run_matrix _prompt_hash), from the
# same Tier-A host records (/data/llm/Strata-tests/tier-a-20261007/full).
EXPECTED_PROMPT_HASH: dict[str, str] = {"short_code": "563cf85e6f7c1247",
                                        "long_doc": "0157820299822d1b"}

# Never start an arm with less than this much of the session budget remaining:
# refuse (exit non-zero, incomplete) rather than launch something that would
# overrun the predeclared session cap.
MIN_REMAINING_S: float = 60.0


@dataclass(frozen=True)
class Bounds:
    """Predeclared limits, not measured results.  Defaults come from
    docs/NIBBLER_TIER_B_PLAN.md 'Predeclare bounds before the first host launch'."""
    per_arm_timeout_s: float = 600.0   # model per-arm timeout at most 600 s
    session_cap_s: float = 3600.0      # combined measurement session cap
    max_windows: int = 16              # matches TierBExport::open(dir, 16, ...) in generate.cpp
    max_export_bytes: int = 64 * 1024 * 1024  # 64 MiB full-logit export per arm (64<<20 in generate.cpp)
    recorded_pairs: int = 3            # three interleaved recorded pairs / family / workload
    max_new_short: int = 256           # short_code budget
    max_new_long: int = 512            # long_doc budget
    # Speculative draft lengths vary run to run (host session 1: identical
    # default configs split a window at pos 8919 as T=4 vs T=3), so windows are
    # matched by (pos, T, input_hash), not ordinal.  At least this many matched
    # windows must be compared bitwise; fewer rejects the pair.
    min_matched_windows: int = 12

    # Hard ceilings: refuse to enlarge past these rather than overrun silently.
    # max_windows/max_export match exactly what the binary enforces at
    # generate.cpp ~11481: TierBExport::open(dir, 16, (int64_t) 64 << 20, ...).
    _CEIL_TIMEOUT_S = 600.0
    _CEIL_SESSION_S = 3600.0
    _CEIL_WINDOWS = 16
    _CEIL_EXPORT = 64 * 1024 * 1024

    def validate(self) -> None:
        checks = [
            ("per_arm_timeout_s", self.per_arm_timeout_s, 0.0, self._CEIL_TIMEOUT_S),
            ("session_cap_s", self.session_cap_s, 0.0, self._CEIL_SESSION_S),
            ("max_windows", self.max_windows, 0, self._CEIL_WINDOWS),
            ("max_export_bytes", self.max_export_bytes, 0, self._CEIL_EXPORT),
        ]
        for name, val, lo, hi in checks:
            if not (val > lo):
                raise ValueError(f"bound {name}={val} must be > {lo}")
            if val > hi:
                raise ValueError(f"bound {name}={val} exceeds hard ceiling {hi}")
        if self.recorded_pairs < 1:
            raise ValueError("recorded_pairs must be >= 1")
        if not (1 <= self.min_matched_windows <= self.max_windows):
            raise ValueError(f"min_matched_windows={self.min_matched_windows} must be in "
                             f"1..max_windows ({self.max_windows})")

    def max_new_for(self, workload: str) -> int:
        return self.max_new_short if workload == "short_code" else self.max_new_long


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT: strip every ambient STRATA_*, then allowlist + one override
# ─────────────────────────────────────────────────────────────────────────────

def ambient_strata_vars(base_env: dict[str, str]) -> tuple[str, ...]:
    """Every ambient STRATA_* name.  Passed to run_arm as controlled_vars so the
    hardened runner strips them all (not just Tier-A's short list) before the
    explicit allowlist is applied."""
    return tuple(sorted(k for k in base_env if k.startswith("STRATA_")))


def assert_single_family_override(overrides: dict[str, str]) -> None:
    """Refuse a combined all-off arm.  At most one family opt-out may be set."""
    present = [v for v in FAMILY_OVERRIDE_VARS if v in overrides]
    if len(present) > 1:
        raise ValueError(
            f"combined all-off arm refused: {present} set together; "
            "each family must be isolated in its own pair"
        )


def build_arm_env(family: Family | None, mode: str) -> dict[str, str]:
    """Explicit STRATA allowlist for one arm.

    Starts from COMMON_STRATA, adds the single family override for the B arm
    (family is None for an A arm), and - only in diagnostic mode - the trace
    control STRATA_TIER_B_TRACE=1.  The export directory is NOT passed via the
    environment: it is supplied once, as an absolute path, through the
    --tier-b-export flag only (dropping the duplicate STRATA_TIER_B_EXPORT so
    there is a single source of truth for the export location).  Timing arms get
    no diagnostic controls at all.
    """
    if mode not in ("diagnostic", "timing"):
        raise ValueError(f"unknown mode {mode!r}")
    env = dict(COMMON_STRATA)
    if family is not None:
        env[family.override_env] = "1"
    if mode == "diagnostic":
        env["STRATA_TIER_B_TRACE"] = "1"
    assert_single_family_override(env)
    return env


def effective_env(base_env: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    """Reproduce the runner's build_env for recording/auditing: strip every
    ambient STRATA_*, then apply overrides.  Mirrors engine_cli.build_env with
    controlled_vars=ambient_strata_vars(base_env)."""
    stripped = tuple(k for k in base_env if k.startswith("STRATA_"))
    env = {k: v for k, v in base_env.items() if k not in stripped}
    env.update(overrides)
    return env


def assert_clean_timing(env_overrides: dict[str, str], flags: list[str]) -> None:
    """Diagnostic data is never a timing sample.  A timing arm must carry no
    trace/export env and no readback/dump flag."""
    bad_env = [v for v in DIAGNOSTIC_VARS if v in env_overrides]
    if bad_env:
        raise ValueError(f"timing arm carries diagnostic env {bad_env}")
    bad_flags = [f for f in DIAGNOSTIC_FLAGS if f in flags]
    if bad_flags:
        raise ValueError(f"timing arm carries diagnostic flag {bad_flags}")


# ─────────────────────────────────────────────────────────────────────────────
# FIXED LAUNCH FLAGS  (Tier-A reference values; Tier-B exe/artifacts only)
# ─────────────────────────────────────────────────────────────────────────────

def fixed_flags(pack: str, native: str, ple: str, expert_profile: str, mtp: str) -> list[str]:
    """Plan 'Fixed model settings': expert-cache 2000, pcie-frac 0, adapt 0,
    prefill auto, spec 4 / spec-min-p 0.5, max-context 262144, kv-resident
    32768, kv int8, vram-reserve 2048, --stats, --stop-eos.  Greedy sampling
    (no --seed).  Values mirror tools/tier_a/_config.py (read-only reference)."""
    return [
        "--pack", pack,
        "--native", native,
        "--ple-gguf", ple,
        "--expert-profile", expert_profile,
        "--mtp", mtp,
        "--expert-cache", "2000",
        "--pcie-frac", "0",
        "--adapt-every", "0",
        "--prefill", "auto",
        "--spec", "4",
        "--spec-min-p", "0.5",
        "--max-context", "262144",
        "--kv-resident", "32768",
        "--kv", "int8",
        "--vram-reserve-mib", "2048",
        "--stats",
        "--stop-eos",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TRACE PARSING  (STRATA_TIER_B_TRACE records on stderr)
# ─────────────────────────────────────────────────────────────────────────────

_TRACE_PREFIX = "TIER_B_TRACE v=1 "


def parse_trace(stderr_text: str) -> list[dict[str, str]]:
    """Parse every complete TIER_B_TRACE v=1 record line into a dict of its
    key=value tokens.  Partial lines (no trailing values) are ignored."""
    records: list[dict[str, str]] = []
    for line in stderr_text.splitlines():
        if not line.startswith(_TRACE_PREFIX):
            continue
        rec: dict[str, str] = {}
        for tok in line[len("TIER_B_TRACE "):].split():
            if "=" not in tok:
                rec = {}
                break
            k, v = tok.split("=", 1)
            rec[k] = v
        if rec.get("v") == "1" and "family" in rec and "path" in rec:
            records.append(rec)
    return records


def trace_truncated(records: list[dict[str, str]]) -> bool:
    return any(r.get("trunc") == "1" or r.get("path") == "truncated" for r in records)


class TraceVerdict(NamedTuple):
    ok: bool          # dispatch evidence present in the untruncated prefix, no forbidden record
    complete: bool    # the recorder did NOT truncate (full dispatch coverage)
    reason: str


# The recorder caps at kMaxRecords (1024) and a cold model run will almost
# certainly truncate.  Truncation is therefore NOT treated as outright failure:
# positive records in the untruncated prefix count as dispatch evidence, while
# trunc=1 only clears `complete` (coverage is explicitly incomplete and is never
# claimed as full).  A forbidden record anywhere in the prefix still fails,
# because what WAS recorded already proves the wrong path ran.  Dispatch is
# rung 2 evidence only; the export pair is the model proof.


def validate_normrope_trace(records: list[dict[str, str]], is_default_arm: bool) -> TraceVerdict:
    """A (default) must dispatch the fused norm+rope kernel; B (opt-out) must
    fall back to the plain rope path and emit NO fused record.  Truncation marks
    the verdict incomplete but, if positive evidence is already present and no
    forbidden record appears in the prefix, does not reject it."""
    trunc = trace_truncated(records)
    complete = not trunc
    fused = [r for r in records if r.get("family") == "rope_norm" and r.get("path") == "norm_rope_fused"]
    plain = [r for r in records if r.get("family") == "rope" and r.get("path") in _ROPE_PLAIN_PATHS]
    if is_default_arm:
        if not fused:
            return TraceVerdict(False, complete, "default arm emitted no norm_rope_fused record in prefix"
                                + (" (inconclusive: recorder truncated first)" if trunc else ""))
        return TraceVerdict(True, complete,
                            f"fused norm+rope dispatched ({len(fused)} records)"
                            + (" [prefix truncated]" if trunc else ""))
    if fused:
        return TraceVerdict(False, complete, "opt-out arm still dispatched norm_rope_fused (forbidden)")
    # rope_inplace32 is selected only when native_norm_rope_usable() is true, which
    # STRATA_NO_NORM_ROPE=1 forces false: seeing it proves the opt-out did not apply.
    if any(r.get("path") == "rope_inplace32" for r in plain):
        return TraceVerdict(False, complete, "opt-out arm dispatched rope_inplace32 (opt-out not applied)")
    if not plain:
        return TraceVerdict(False, complete, "opt-out arm emitted no plain rope dispatch in prefix")
    return TraceVerdict(True, complete,
                        f"plain rope dispatched ({len(plain)} records)"
                        + (" [prefix truncated]" if trunc else ""))


def _sub16_records(records: list[dict[str, str]], caller: str, geom_key: str, geom_val: int):
    out = []
    for r in records:
        if r.get("family") != "iq_grouped" or r.get("caller") != caller:
            continue
        if r.get(geom_key) != str(geom_val):
            continue
        out.append(r)
    return out


def validate_sub16_trace(records: list[dict[str, str]], is_default_arm: bool) -> TraceVerdict:
    """A (default) must dispatch split_sub16 for BOTH GU (n_embd=2560) and down
    (n_ff=640); B (opt-out) must dispatch a reference layout and NO split_sub16.
    Only positive, supported records at the studied geometry count;
    skipped/unsupported/zero/truncated/format_error never count.  Truncation of
    the recorder marks the verdict incomplete without rejecting present
    evidence."""
    trunc = trace_truncated(records)
    complete = not trunc
    gu = _sub16_records(records, "native_expert_grouped.gu", "n_embd", SUB16_N_EMBD)
    dn = _sub16_records(records, "native_expert_grouped.down", "n_ff", SUB16_N_FF)
    gu = [r for r in gu if r.get("path") not in _NONPOSITIVE_PATHS]
    dn = [r for r in dn if r.get("path") not in _NONPOSITIVE_PATHS]
    gu_sub = [r for r in gu if r.get("path") == _SUB16_PATH]
    dn_sub = [r for r in dn if r.get("path") == _SUB16_PATH]
    if is_default_arm:
        # Forbidden-record check first: a reference dispatch in the default arm
        # is proof the wrong path ran, independent of truncation.
        if not gu_sub or not dn_sub:
            return TraceVerdict(False, complete, "default arm missing split_sub16 on GU and/or down in prefix"
                                + (" (inconclusive: recorder truncated first)" if trunc else ""))
        return TraceVerdict(True, complete,
                            f"split_sub16 GU={len(gu_sub)} down={len(dn_sub)}"
                            + (" [prefix truncated]" if trunc else ""))
    if gu_sub or dn_sub:
        return TraceVerdict(False, complete, "opt-out arm still dispatched split_sub16 (forbidden)")
    gu_ref = [r for r in gu if r.get("path") in _SUB16_REF_PATHS]
    dn_ref = [r for r in dn if r.get("path") in _SUB16_REF_PATHS]
    if not gu_ref or not dn_ref:
        return TraceVerdict(False, complete, "opt-out arm emitted no reference GU/down dispatch in prefix")
    return TraceVerdict(True, complete,
                        f"reference GU={len(gu_ref)} down={len(dn_ref)}"
                        + (" [prefix truncated]" if trunc else ""))


def validate_trace(records: list[dict[str, str]], family: Family, is_default_arm: bool) -> TraceVerdict:
    if family.name == "normrope":
        return validate_normrope_trace(records, is_default_arm)
    if family.name == "sub16":
        return validate_sub16_trace(records, is_default_arm)
    return TraceVerdict(False, False, f"unknown family {family.name}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT PARSING  (sta-osh.5 manifest.txt + rows.bin)
# ─────────────────────────────────────────────────────────────────────────────

_ROWS_MAGIC = b"TBE1"
_ROWS_HEADER = struct.Struct("<4sIqqq")  # magic, version, vocab, row_floats, reserved
_ROWS_HEADER_BYTES = 32
# Verifier::run accepts 1..max_t; init caps max_t at kVerifyMaxT and cpu::MAXT,
# both 8 (include/strata/kernels/verify_kernels.hpp and cpu/expert.hpp).
VERIFY_MAX_T = 8

_W_RE = re.compile(
    r"^W (\d+) (\d+) (\d+) (\d+) (\d+) "
    r"([0-9a-f]{16}) ([0-9a-f]{16}) (-?\d+) (-?\d+) (\S+) natural picks=(.*)$"
)


@dataclass
class Window:
    ordinal: int
    pos: int
    T: int
    accepted: int
    emitted: int
    input_hash: str
    window_hash: str
    rows_offset: int
    rows_bytes: int
    status: str
    picks: list[int]


@dataclass
class ExportManifest:
    vocab: int = -1
    max_windows: int = -1
    max_row_bytes: int = -1
    windows: list[Window] = field(default_factory=list)
    truncation: bool = False
    export_error: str | None = None
    complete: bool = False
    parse_error: str | None = None


def parse_manifest(text: str) -> ExportManifest:
    """Parse exporter metadata, requiring canonical, unique terminal lines.

    finish() writes optional truncation, then either an export error or a
    successful completion footer. No window/header data can follow these lines.
    Error IDs may contain spaces (e.g. "rows close").
    """
    m = ExportManifest()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "# tier-b native window export v=1":
        m.parse_error = "missing/stale manifest header"
        return m
    seen_terminal: set[str] = set()
    for line in lines[1:]:
        # Recognize malformed spellings/spacing too, so a later valid footer
        # cannot hide them. Only the exporter's exact success spelling counts.
        terminal = re.match(r"^#\s*(truncation|export_error|complete)", line.strip())
        if terminal:
            key = terminal.group(1)
            if key in seen_terminal:
                m.parse_error = f"duplicate terminal metadata: {key}"
                return m
            if m.complete or (key == "truncation" and m.export_error is not None):
                m.parse_error = f"terminal metadata out of order: {line!r}"
                return m
            seen_terminal.add(key)
            if key == "export_error":
                prefix = "# export_error "
                error_id = line[len(prefix):] if line.startswith(prefix) else ""
                if not error_id or error_id != error_id.strip():
                    m.parse_error = f"malformed export_error line: {line!r}"
                    return m
                m.export_error = error_id
            elif line != f"# {key} 1":
                m.parse_error = f"malformed {key} line: {line!r}"
                return m
            elif key == "complete":
                if m.export_error is not None:
                    m.parse_error = "complete footer conflicts with export_error"
                    return m
                m.complete = True
            else:
                m.truncation = True
            continue
        if seen_terminal and line.strip():
            m.parse_error = f"data after terminal metadata: {line!r}"
            return m
        if line.startswith("# vocab "):
            m.vocab = int(line.split()[2])
        elif line.startswith("# max_windows "):
            m.max_windows = int(line.split()[2])
        elif line.startswith("# max_row_bytes "):
            m.max_row_bytes = int(line.split()[2])
        elif line.startswith("W "):
            mm = _W_RE.match(line)
            if mm is None:
                m.parse_error = f"malformed W line: {line!r}"
                return m
            picks_raw = mm.group(11).strip()
            try:
                picks = [int(x) for x in picks_raw.split(",")] if picks_raw else []
            except ValueError:
                m.parse_error = f"bad picks in: {line!r}"
                return m
            m.windows.append(Window(
                ordinal=int(mm.group(1)), pos=int(mm.group(2)), T=int(mm.group(3)),
                accepted=int(mm.group(4)), emitted=int(mm.group(5)),
                input_hash=mm.group(6), window_hash=mm.group(7),
                rows_offset=int(mm.group(8)), rows_bytes=int(mm.group(9)),
                status=mm.group(10), picks=picks,
            ))
    return m


class RowsFile(NamedTuple):
    vocab: int
    row_floats: int
    n_rows: int
    data: bytes          # raw float payload AFTER the 32-byte header
    error: str | None


def parse_rows(blob: bytes) -> RowsFile:
    if len(blob) < _ROWS_HEADER_BYTES:
        return RowsFile(-1, -1, 0, b"", "rows.bin shorter than header")
    magic, ver, vocab, row_floats, _rsv = _ROWS_HEADER.unpack(blob[:_ROWS_HEADER_BYTES])
    if magic != _ROWS_MAGIC:
        return RowsFile(-1, -1, 0, b"", "rows.bin bad magic")
    if ver != 1:
        return RowsFile(-1, -1, 0, b"", f"rows.bin bad version {ver}")
    if vocab <= 0 or row_floats != vocab:
        return RowsFile(vocab, row_floats, 0, b"", "rows.bin vocab/row_floats mismatch")
    payload = blob[_ROWS_HEADER_BYTES:]
    row_bytes = vocab * 4
    if len(payload) % row_bytes != 0:
        return RowsFile(vocab, row_floats, 0, payload, "rows.bin payload not a whole number of rows")
    return RowsFile(vocab, row_floats, len(payload) // row_bytes, payload, None)


class ExportVerdict(NamedTuple):
    ok: bool                 # successful finalization AND valid nonempty windows
    full_proof: bool         # ok AND not truncated
    n_windows: int
    reason: str


def validate_export(manifest: ExportManifest, rows: RowsFile,
                    bounds: Bounds | None = None) -> ExportVerdict:
    """Require successful finalization and valid rows/spans/finite values/shape.
    Any export error or missing/invalid completion footer rejects all proof.
    Only a successfully finalized budget-truncated export may supply bounded
    prefix evidence (never full proof). Metadata/footer alone is insufficient:
    at least one valid 'ok' window with real rows is required.

    When `bounds` is supplied, the manifest's declared budget header must match
    what the binary enforces: `# max_windows` / `# max_row_bytes` must equal the
    predeclared bounds, and the number of recorded windows must not exceed
    bounds.max_windows.  A header that disagrees means the harness and binary
    bounds have drifted and the export is rejected."""
    if manifest.parse_error:
        return ExportVerdict(False, False, 0, f"manifest: {manifest.parse_error}")
    if manifest.export_error is not None:
        return ExportVerdict(False, False, 0, f"export_error={manifest.export_error}")
    if not manifest.complete:
        return ExportVerdict(False, False, 0, "no complete footer (generation/export unfinished)")
    if rows.error:
        return ExportVerdict(False, False, 0, f"rows: {rows.error}")
    if manifest.vocab != rows.vocab:
        return ExportVerdict(False, False, 0,
                             f"vocab mismatch manifest={manifest.vocab} rows={rows.vocab}")
    if bounds is not None:
        if manifest.max_windows != bounds.max_windows:
            return ExportVerdict(False, False, 0,
                                 f"manifest max_windows={manifest.max_windows} != declared bound"
                                 f" {bounds.max_windows}")
        if manifest.max_row_bytes != bounds.max_export_bytes:
            return ExportVerdict(False, False, 0,
                                 f"manifest max_row_bytes={manifest.max_row_bytes} != declared bound"
                                 f" {bounds.max_export_bytes}")
        if len(manifest.windows) > bounds.max_windows:
            return ExportVerdict(False, False, len(manifest.windows),
                                 f"window count {len(manifest.windows)} exceeds bound"
                                 f" {bounds.max_windows}")
    ok_windows = 0
    spans: list[tuple[int, int]] = []
    last_ordinal = max((w.ordinal for w in manifest.windows), default=-1)
    row_bytes = manifest.vocab * 4
    for w in manifest.windows:
        if w.status != "ok":
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} non-ok status {w.status!r}")
        if not (1 <= w.T <= VERIFY_MAX_T):
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} unsupported T={w.T} (required 1..{VERIFY_MAX_T})")
        # generate.cpp commits a+1 inputs, then emits outv[0..a] subject to
        # max_new/EOS. The loop starts with budget remaining, so emitted is
        # positive, but need NOT equal accepted+1 (nor the T verifier picks).
        if not (0 <= w.accepted < w.T and 1 <= w.emitted <= w.accepted + 1):
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} bad accepted/emitted counts "
                                 f"accepted={w.accepted} emitted={w.emitted} T={w.T}")
        if w.emitted < w.accepted + 1 and w.ordinal != last_ordinal:
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} clipped emitted prefix before later window")
        if len(w.picks) != w.T:
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} picks/T mismatch")
        if any(p < 0 or p >= manifest.vocab for p in w.picks):
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} verifier pick outside vocab")
        want = w.T * row_bytes
        if w.rows_bytes <= 0 or w.rows_bytes != want or w.rows_offset < _ROWS_HEADER_BYTES:
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} bad row span")
        start = w.rows_offset - _ROWS_HEADER_BYTES
        end = start + w.rows_bytes
        if start % row_bytes != 0:
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} row span not row-aligned")
        if end > len(rows.data):
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} rows beyond file (short export)")
        if any(start < prev_end and prev_start < end for prev_start, prev_end in spans):
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} overlapping row span")
        spans.append((start, end))
        floats = struct.unpack(f"<{w.T * manifest.vocab}f", rows.data[start:end])
        if not all(math.isfinite(x) for x in floats):
            return ExportVerdict(False, False, ok_windows,
                                 f"window {w.ordinal} non-finite logits")
        ok_windows += 1
    if ok_windows == 0:
        return ExportVerdict(False, False, 0, "no completed windows (metadata/footer alone)")
    full = not manifest.truncation
    reason = "full proof" if full else "coverage incomplete: truncated (successfully finalized)"
    return ExportVerdict(True, full, ok_windows, reason)


# ─────────────────────────────────────────────────────────────────────────────
# PAIR CORRECTNESS: matched windows + predeclared contract on logit rows
# ─────────────────────────────────────────────────────────────────────────────

def window_payload(rows: RowsFile, w: Window) -> bytes:
    """Raw float32 bytes for a window whose span validate_export has checked."""
    start = w.rows_offset - _ROWS_HEADER_BYTES
    return rows.data[start:start + w.rows_bytes]


def window_floats(manifest: ExportManifest, rows: RowsFile, w: Window) -> tuple[float, ...]:
    return struct.unpack(f"<{w.T * manifest.vocab}f", window_payload(rows, w))


class PairVerdict(NamedTuple):
    ok: bool
    full_proof: bool
    reason: str
    n_matched: int = 0
    n_unmatched_a: int = 0
    n_unmatched_b: int = 0


def window_key(w: Window) -> tuple[int, int, str]:
    """Identity of one verifier computation: same start position, same window
    length and the same input token IDs (fnv1a over the T int32 inputs)."""
    return (w.pos, w.T, w.input_hash)


def compare_pair(family: Family,
                 man_a: ExportManifest, rows_a: RowsFile,
                 man_b: ExportManifest, rows_b: RowsFile,
                 bounds: Bounds | None = None) -> PairVerdict:
    """Controlled A/B comparison over windows matched by (pos, T, input_hash).

    Speculative draft lengths are not deterministic run to run, so the two arms
    may window the same committed tokens differently; ordinal alignment is not
    a valid contract.  Both arms must export valid windows with unique keys.
    For every key present in BOTH arms accepted/emitted counts, the emitted
    prefix IDs and all T verifier picks must be equal, and the predeclared contract (bitwise for both first-wave identity families;
    the tolerance branch exists only for future declared families) is applied
    to the logit rows.  At least `min_matched_windows` (bounds; 1 without
    bounds) keys must match.  Unmatched windows are counted, never compared and
    never proof: any unmatched window makes coverage incomplete (no full proof).
    Whole-run token identity is enforced separately by the caller."""
    va = validate_export(man_a, rows_a, bounds)
    vb = validate_export(man_b, rows_b, bounds)
    if not va.ok:
        return PairVerdict(False, False, f"arm A export invalid: {va.reason}")
    if not vb.ok:
        return PairVerdict(False, False, f"arm B export invalid: {vb.reason}")
    keyed: list[dict[tuple[int, int, str], Window]] = []
    for arm, man in (("A", man_a), ("B", man_b)):
        d: dict[tuple[int, int, str], Window] = {}
        for w in man.windows:
            k = window_key(w)
            if k in d:
                return PairVerdict(False, False, f"arm {arm} duplicate window key {k} "
                                                 f"(ordinals {d[k].ordinal}, {w.ordinal})")
            d[k] = w
        keyed.append(d)
    ka, kb = keyed
    common = sorted(set(ka) & set(kb))
    n_ua, n_ub = len(ka) - len(common), len(kb) - len(common)
    min_matched = bounds.min_matched_windows if bounds is not None else 1
    if len(common) < min_matched:
        return PairVerdict(False, False, f"matched windows {len(common)} < required {min_matched} "
                                         f"(unmatched A={n_ua} B={n_ub})", len(common), n_ua, n_ub)
    for k in common:
        wa, wb = ka[k], kb[k]
        where = f"window pos={k[0]} T={k[1]}"
        if (wa.accepted, wa.emitted) != (wb.accepted, wb.emitted):
            return PairVerdict(False, False, f"accepted/emitted counts mismatch at {where}",
                               len(common), n_ua, n_ub)
        if wa.picks[:wa.emitted] != wb.picks[:wb.emitted]:
            return PairVerdict(False, False, f"emitted token IDs mismatch at {where}",
                               len(common), n_ua, n_ub)
        if wa.picks != wb.picks:
            return PairVerdict(False, False, f"verifier picks mismatch at {where}",
                               len(common), n_ua, n_ub)
    # Contract on the logit rows themselves.  Both exports have independently
    # passed finite-value validation above; equal bytes alone are not enough.
    for k in common:
        wa, wb = ka[k], kb[k]
        payload_a = window_payload(rows_a, wa)
        payload_b = window_payload(rows_b, wb)
        where = f"window pos={k[0]} T={k[1]} (ordinals A={wa.ordinal} B={wb.ordinal})"
        if len(payload_a) != len(payload_b):
            return PairVerdict(False, False, f"logit length mismatch at {where}",
                               len(common), n_ua, n_ub)
        if family.contract == "bitwise":
            # Python float equality collapses +0.0/-0.0; compare the original
            # float32 encodings, never decoded or repacked values.
            if payload_a != payload_b:
                return PairVerdict(False, False, f"bitwise contract violated at {where}",
                                   len(common), n_ua, n_ub)
        else:
            fa = window_floats(man_a, rows_a, wa)
            fb = window_floats(man_b, rows_b, wb)
            for x, y in zip(fa, fb):
                denom = max(abs(x), abs(y), 1.0)
                if abs(x - y) / denom > family.rel_tol:
                    return PairVerdict(False, False, f"tolerance {family.rel_tol:g} exceeded at {where}",
                                       len(common), n_ua, n_ub)
    full = va.full_proof and vb.full_proof and n_ua == 0 and n_ub == 0
    reason = (f"{len(common)} matched windows bitwise" if family.contract == "bitwise"
              else f"{len(common)} matched windows within tolerance")
    if not full:
        reason += f"; coverage incomplete (unmatched A={n_ua} B={n_ub}, export truncation)"
    return PairVerdict(True, full, reason, len(common), n_ua, n_ub)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC EVALUATION: read a pair's artifacts and produce a verdict
# ─────────────────────────────────────────────────────────────────────────────

def read_export_dir(export_dir: str) -> tuple[ExportManifest, RowsFile]:
    """Read manifest.txt + rows.bin from an arm's --tier-b-export directory and
    parse them with the real parsers.  Missing files surface as parse/row errors
    rather than exceptions."""
    mp = Path(export_dir) / "manifest.txt"
    rp_ = Path(export_dir) / "rows.bin"
    try:
        man = parse_manifest(mp.read_text(encoding="utf-8"))
    except OSError as exc:
        man = ExportManifest(parse_error=f"cannot read manifest.txt: {exc}")
    try:
        rows = parse_rows(rp_.read_bytes())
    except OSError as exc:
        rows = RowsFile(-1, -1, 0, b"", f"cannot read rows.bin: {exc}")
    return man, rows


def evaluate_diagnostic_pair(family: Family, workload: str,
                             res_a: ArmResult, res_b: ArmResult,
                             export_dir_a: str, export_dir_b: str,
                             bounds: Bounds) -> dict:
    """Full diagnostic verdict for one A/B pair.

    Requires both arms to exit 0 with no runner error, parses + validates each
    arm's trace (dispatch evidence, rung 2) and export (rows, rung 3), then runs
    the controlled compare_pair under the family contract (the model proof).
    Returns a JSON-serialisable verdict dict; `proof` is the pair gate and
    `full_proof` records whether coverage was complete (untruncated) on both
    rungs."""
    reasons: list[str] = []

    # Rung 1: both arms must run cleanly.
    arm_errors = {"A": res_a.error, "B": res_b.error}
    clean = all(e is None for e in arm_errors.values())
    if not clean:
        reasons.append(f"arm error(s): {arm_errors}")

    # Rung 2: dispatch trace evidence per arm.
    recs_a = parse_trace(res_a.stderr_raw)
    recs_b = parse_trace(res_b.stderr_raw)
    tv_a = validate_trace(recs_a, family, is_default_arm=True)
    tv_b = validate_trace(recs_b, family, is_default_arm=False)
    if not tv_a.ok:
        reasons.append(f"A trace: {tv_a.reason}")
    if not tv_b.ok:
        reasons.append(f"B trace: {tv_b.reason}")

    # Rung 3: export validity per arm.
    man_a, rows_a = read_export_dir(export_dir_a)
    man_b, rows_b = read_export_dir(export_dir_b)
    ev_a = validate_export(man_a, rows_a, bounds)
    ev_b = validate_export(man_b, rows_b, bounds)
    if not ev_a.ok:
        reasons.append(f"A export: {ev_a.reason}")
    if not ev_b.ok:
        reasons.append(f"B export: {ev_b.reason}")

    # Model proof: controlled A/B comparison under the family contract.
    pv = compare_pair(family, man_a, rows_a, man_b, rows_b, bounds)
    if not pv.ok:
        reasons.append(f"pair: {pv.reason}")

    # Whole-run identity: the export covers a bounded prefix of windows, but the
    # parsed prompt echo and emitted IDs cover the entire run.  Both families are
    # identity arms, so any difference rejects the controlled comparison.
    ids_match = (res_a.prompt_ids == res_b.prompt_ids and bool(res_a.output_ids)
                 and res_a.output_ids == res_b.output_ids)
    if not ids_match:
        reasons.append("prompt/output token IDs differ between arms over the whole run")

    proof = clean and tv_a.ok and tv_b.ok and ev_a.ok and ev_b.ok and pv.ok and ids_match
    full_proof = proof and tv_a.complete and tv_b.complete and pv.full_proof
    n_windows = pv.n_matched
    if proof and not reasons:
        reasons.append("controlled correctness proven"
                       + ("" if full_proof else " (coverage incomplete: truncated prefix/windows)"))
    return {
        "family": family.name, "workload": workload,
        "proof": proof, "full_proof": full_proof, "n_windows": n_windows,
        "arms_clean": clean, "arm_errors": arm_errors,
        "trace": {"A": {"ok": tv_a.ok, "complete": tv_a.complete, "reason": tv_a.reason},
                  "B": {"ok": tv_b.ok, "complete": tv_b.complete, "reason": tv_b.reason}},
        "export": {"A": {"ok": ev_a.ok, "full_proof": ev_a.full_proof,
                         "n_windows": ev_a.n_windows, "reason": ev_a.reason},
                   "B": {"ok": ev_b.ok, "full_proof": ev_b.full_proof,
                         "n_windows": ev_b.n_windows, "reason": ev_b.reason}},
        "pair": {"ok": pv.ok, "full_proof": pv.full_proof, "reason": pv.reason,
                 "n_matched": pv.n_matched, "n_unmatched": {"A": pv.n_unmatched_a,
                                                            "B": pv.n_unmatched_b}},
        "whole_run_ids_match": ids_match,
        "output_hash": {"A": res_a.output_hash, "B": res_b.output_hash},
        "n_output": {"A": res_a.n_output, "B": res_b.n_output},
        "reasons": reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TIMING: require primary metrics with valid n; keep raw ordered paired samples
# ─────────────────────────────────────────────────────────────────────────────

class TimingSample(NamedTuple):
    ok: bool
    decode_tok_s: float
    prefill_tok_s: float
    n_decode: int
    reason: str


def timing_sample(result: ArmResult, n_prompt: int) -> TimingSample:
    """A clean-timing arm must have a parser-valid decode (count matching
    output, positive finite tok/s) and, for a multi-token prompt, valid prefill.
    Diagnostic durations are never timings, so a run with any error is rejected."""
    if result.error not in (None,):
        return TimingSample(False, 0.0, 0.0, 0, f"arm error: {result.error}")
    if result.n_output <= 0:
        return TimingSample(False, 0.0, 0.0, 0, "no decode tokens")
    if not (math.isfinite(result.decode_tok_s) and result.decode_tok_s > 0):
        return TimingSample(False, 0.0, 0.0, 0, "non-positive decode tok/s")
    if n_prompt > 1 and not (math.isfinite(result.prefill_tok_s) and result.prefill_tok_s > 0):
        return TimingSample(False, 0.0, 0.0, 0, "non-positive prefill tok/s on multi-token prompt")
    return TimingSample(True, result.decode_tok_s, result.prefill_tok_s, result.n_output, "ok")


def summarise_pairs(samples_a: list[float], samples_b: list[float]) -> dict:
    """Paired deltas/ratios, per-arm median and spread (range + stdev), n.
    Three pairs are a bounded minimum, not statistical significance."""
    n = min(len(samples_a), len(samples_b))
    a = samples_a[:n]
    b = samples_b[:n]
    deltas = [y - x for x, y in zip(a, b)]
    ratios = [y / x for x, y in zip(a, b) if x]

    def spread(xs: list[float]) -> dict:
        if not xs:
            return {"median": None, "range": None, "stdev": None, "n": 0}
        return {
            "median": statistics.median(xs),
            "range": max(xs) - min(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "n": len(xs),
        }

    return {
        "n_pairs": n,
        "raw_a": a, "raw_b": b, "paired_deltas": deltas, "paired_ratios": ratios,
        "arm_a": spread(a), "arm_b": spread(b),
        "note": "cold fresh-process CLI decode (includes graph capture); "
                "not warmed/steady-state; three pairs are a bounded minimum.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ARM / PAIR PLANNING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArmSpec:
    label: str
    family: str
    workload: str
    arm: str            # "A" (default) or "B" (opt-out)
    mode: str           # "diagnostic" | "timing"
    recorded: bool      # False for the excluded preparatory timing process
    round_idx: int      # -1 for diagnostic / preparatory, else recorded round
    env_overrides: dict[str, str]
    flags: list[str]
    export_dir: str | None     # absolute export path (diagnostic only)


@dataclass
class Pair:
    family: Family
    workload: str
    recorded: bool
    round_idx: int             # -1 for diagnostic / preparatory
    arm_a: ArmSpec             # default arm
    arm_b: ArmSpec             # opt-out arm
    order: tuple[str, str]     # launch order ("A","B") or ("B","A")

    def launch_specs(self) -> list[ArmSpec]:
        by = {"A": self.arm_a, "B": self.arm_b}
        return [by[self.order[0]], by[self.order[1]]]


def _counterbalance(round_idx: int) -> tuple[str, str]:
    """AB on even rounds, BA on odd rounds (plan: alternate AB/BA/AB)."""
    return ("A", "B") if round_idx % 2 == 0 else ("B", "A")


def _make_arm(family: Family, workload: str, arm: str, mode: str, recorded: bool,
              round_idx: int, base_flags: list[str], out_dir: Path | None) -> ArmSpec:
    """Build one ArmSpec.  A = default (no override), B = opt-out (one family var).

    Diagnostic arms get STRATA_TIER_B_TRACE=1 and an ABSOLUTE, not-yet-existing
    --tier-b-export directory (the exporter refuses an existing dir, and the
    child cwd is the harness cwd so a relative path would be wrong).  Timing arms
    carry no diagnostics at all."""
    fam = None if arm == "A" else family
    tag = "diag" if mode == "diagnostic" else ("prep" if not recorded else f"rec{round_idx}")
    label = f"{family.name}.{workload}.{tag}.{arm}"
    flags = list(base_flags)
    export_dir: str | None = None
    if mode == "diagnostic":
        if out_dir is None:
            raise ValueError("diagnostic mode requires an --out directory for export artifacts")
        export_dir = str((out_dir / "arms" / label / "export").resolve())
        flags = flags + ["--tier-b-export", export_dir]
    env = build_arm_env(fam, mode)
    if mode == "timing":
        assert_clean_timing(env, flags)
    return ArmSpec(
        label=label, family=family.name, workload=workload, arm=arm, mode=mode,
        recorded=recorded, round_idx=round_idx, env_overrides=env, flags=flags,
        export_dir=export_dir,
    )


def plan_diagnostic(families: list[Family], workloads: list[str],
                    base_flags: list[str], out_dir: Path | None) -> list[Pair]:
    """One A/B pair per family/workload, trace+export on, launch order A then B."""
    pairs: list[Pair] = []
    for fam in families:
        for wl in workloads:
            a = _make_arm(fam, wl, "A", "diagnostic", True, -1, base_flags, out_dir)
            b = _make_arm(fam, wl, "B", "diagnostic", True, -1, base_flags, out_dir)
            pairs.append(Pair(fam, wl, True, -1, a, b, ("A", "B")))
    return pairs


def plan_timing(families: list[Family], workloads: list[str], bounds: Bounds,
                base_flags: list[str]) -> list[Pair]:
    """Interleaved, counterbalanced timing plan with diagnostics OFF.

    Order:
      1. one EXCLUDED preparatory pair per (family, workload) cell, in natural
         cell order (warms disk cache / allocator before any recorded sample);
      2. for each round r in 0..recorded_pairs-1, run recorded pair r for EVERY
         cell, iterating cells in a round-dependent ROTATION so the same family
         is not always launched first; AB/BA/AB counterbalancing per round.
    """
    cells = [(fam, wl) for fam in families for wl in workloads]
    pairs: list[Pair] = []
    # (1) excluded preparatory pair per cell.
    for fam, wl in cells:
        a = _make_arm(fam, wl, "A", "timing", False, -1, base_flags, None)
        b = _make_arm(fam, wl, "B", "timing", False, -1, base_flags, None)
        pairs.append(Pair(fam, wl, False, -1, a, b, ("A", "B")))
    # (2) recorded rounds with per-round cell rotation + counterbalance.
    for r in range(bounds.recorded_pairs):
        rot = r % len(cells) if cells else 0
        rotated = cells[rot:] + cells[:rot]
        order = _counterbalance(r)
        for fam, wl in rotated:
            a = _make_arm(fam, wl, "A", "timing", True, r, base_flags, None)
            b = _make_arm(fam, wl, "B", "timing", True, r, base_flags, None)
            pairs.append(Pair(fam, wl, True, r, a, b, order))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts(dry_run: bool) -> dict[str, list[int]]:
    """Token id lists.  Dry-run uses synthetic ids (no tokenizer/regex).  Live
    runs reuse the fixed Tier-A corpus and verify that each workload's token-id
    count equals the predeclared EXPECTED_PROMPT_IDS from the Tier-A host
    records; a mismatch (stale corpus / changed tokenizer) refuses the run so the
    compared inputs can never drift silently."""
    if dry_run:
        return {"short_code": list(range(1, EXPECTED_PROMPT_IDS["short_code"] + 1)),
                "long_doc": list(range(1, EXPECTED_PROMPT_IDS["long_doc"] + 1))}
    from tier_a.prompts import LONG_DOC_Q, SHORT_CODE  # noqa: PLC0415
    from tier_a.tokenize_prompt import encode_chat, load_tokenizer  # noqa: PLC0415
    from tier_a._config import TOKENIZER_DIR  # noqa: PLC0415
    tok = load_tokenizer(TOKENIZER_DIR)
    prompts = {"short_code": encode_chat(tok, SHORT_CODE), "long_doc": encode_chat(tok, LONG_DOC_Q)}
    assert_prompts_expected(prompts, list(prompts))
    return prompts


def assert_prompts_expected(prompts: dict[str, list[int]], workloads: list[str]) -> None:
    """Verify required workloads encode to the predeclared id counts AND hashes
    (Tier-A host records) before any live launch.  Raises on mismatch."""
    for wl in workloads:
        ids = prompts[wl]
        if len(ids) != EXPECTED_PROMPT_IDS[wl]:
            raise RuntimeError(f"prompt {wl!r} has {len(ids)} ids, expected "
                               f"{EXPECTED_PROMPT_IDS[wl]}; refusing run")
        if prompt_hash(ids) != EXPECTED_PROMPT_HASH[wl]:
            raise RuntimeError(f"prompt {wl!r} hash {prompt_hash(ids)} != expected "
                               f"{EXPECTED_PROMPT_HASH[wl]}; refusing run")


def prompt_hash(ids: list[int]) -> str:
    import hashlib
    return hashlib.sha256(" ".join(str(i) for i in ids).encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# DRIVER
# ─────────────────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str | None:
    """Full sha256 of a file's bytes, or None if unreadable (absent offline, or a
    directory).  Never raises."""
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _stat_identity(path: str) -> dict | None:
    """Cheap identity (size + mtime_ns) for multi-GB model inputs: hashing them
    would stream tens of GB through the page cache right before timing."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


MODEL_INPUT_KEYS: tuple[str, ...] = ("pack", "native", "ple_gguf", "expert_profile", "mtp")
_MODEL_INPUT_KIND = {"pack": "dir", "native": "file", "ple_gguf": "file",
                     "expert_profile": "file", "mtp": "dir"}


def validate_live_inputs(args) -> list[str]:
    """Every live launch input must be a non-empty path of the right kind.
    Returns problems (empty == ok).  Dry-run does not call this."""
    probs: list[str] = []
    if not args.exe or not os.path.isfile(args.exe) or not os.access(args.exe, os.X_OK):
        probs.append(f"--exe not an executable file: {args.exe!r}")
    for k in MODEL_INPUT_KEYS:
        v = getattr(args, k)
        flag = "--" + k.replace("_", "-")
        if not v:
            probs.append(f"{flag} is empty")
        elif _MODEL_INPUT_KIND[k] == "dir" and not os.path.isdir(v):
            probs.append(f"{flag} not a directory: {v}")
        elif _MODEL_INPUT_KIND[k] == "file" and not os.path.isfile(v):
            probs.append(f"{flag} not a file: {v}")
    return probs


def write_provenance(out_dir: Path | None, args, bounds: Bounds,
                     families: list[Family], workloads: list[str],
                     prompts: dict[str, list[int]], base_env: dict[str, str]) -> dict:
    """Run-level manifest: exe sha256, model-input stat identity, source commit,
    bounds, contracts, controlled STRATA vars and prompt provenance."""
    def git(*a: str) -> str | None:
        import subprocess
        try:
            return subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True,
                                  text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    model_inputs = {k: getattr(args, k) for k in MODEL_INPUT_KEYS}
    prov = {
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "exe": args.exe,
        "exe_sha256": _file_hash(args.exe),
        "model_inputs": model_inputs,
        "model_input_identity": {k: _stat_identity(v) for k, v in model_inputs.items() if v},
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain")),
        "families": {f.name: {"override_env": f.override_env, "contract": f.contract,
                              "rel_tol": f.rel_tol} for f in families},
        "workloads": workloads,
        "bounds": {"per_arm_timeout_s": bounds.per_arm_timeout_s,
                   "session_cap_s": bounds.session_cap_s,
                   "max_windows": bounds.max_windows,
                   "max_export_bytes": bounds.max_export_bytes,
                   "recorded_pairs": bounds.recorded_pairs,
                   "min_matched_windows": bounds.min_matched_windows,
                   "max_new_short": bounds.max_new_short,
                   "max_new_long": bounds.max_new_long},
        "common_strata": COMMON_STRATA,
        "controlled_strata_vars": list(ambient_strata_vars(base_env)),
        "prompts": {wl: {"n_ids": len(ids), "hash": prompt_hash(ids)}
                    for wl, ids in prompts.items()},
    }
    if out_dir is not None:
        (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return prov


def _persist(out_dir: Path | None, spec: ArmSpec, result: ArmResult,
             base_env: dict[str, str], timeout_s: float) -> None:
    if out_dir is None:
        return
    d = out_dir / "arms" / spec.label
    d.mkdir(parents=True, exist_ok=True)
    (d / "argv.txt").write_text("\n".join(result.argv), encoding="utf-8")
    (d / "stdout.txt").write_text(result.stdout_raw, encoding="utf-8")
    (d / "stderr.txt").write_text(result.stderr_raw, encoding="utf-8")
    (d / "prompt_ids.txt").write_text(" ".join(map(str, result.prompt_ids)), encoding="utf-8")
    (d / "output_ids.txt").write_text(" ".join(map(str, result.output_ids)), encoding="utf-8")
    eff = effective_env(base_env, spec.env_overrides)
    (d / "env.json").write_text(json.dumps(
        {k: v for k, v in sorted(eff.items()) if k.startswith("STRATA_")}, indent=2
    ), encoding="utf-8")
    (d / "arm.json").write_text(json.dumps({
        "label": spec.label, "family": spec.family, "workload": spec.workload,
        "arm": spec.arm, "mode": spec.mode, "recorded": spec.recorded,
        "round_idx": spec.round_idx, "env_overrides": spec.env_overrides,
        "export_dir": spec.export_dir, "timeout_s": timeout_s,
        "error": result.error, "exit_code": result.exit_code, "wall_s": result.wall_s,
        "n_output": result.n_output, "output_hash": result.output_hash,
        "decode_tok_s": result.decode_tok_s, "prefill_tok_s": result.prefill_tok_s,
        "ttft_ms": result.ttft_ms, "draft_rate": result.draft_rate,
    }, indent=2), encoding="utf-8")


class SessionClock:
    """Session cap enforced PER ARM, not only between arms: each launch gets
    timeout = min(per-arm bound, remaining budget - grace) and an arm is refused
    when less than MIN_REMAINING_S remains, so no single arm can carry the
    session past its cap."""
    GRACE_S = 15.0   # runner SIGKILL + reap + persist after a timeout

    def __init__(self, bounds: Bounds, now: Callable[[], float] = time.monotonic):
        self.bounds = bounds
        self.now = now
        self.t0 = now()

    def elapsed(self) -> float:
        return self.now() - self.t0

    def arm_timeout(self) -> float | None:
        """Timeout for the next arm, or None when it must not be launched."""
        remaining = self.bounds.session_cap_s - self.elapsed() - self.GRACE_S
        if remaining < MIN_REMAINING_S:
            return None
        return min(self.bounds.per_arm_timeout_s, remaining)


def variant_name(family: Family, arm: str) -> str:
    return (f"{family.name}.default" if arm == "A"
            else f"{family.name}.{family.override_env.removeprefix('STRATA_').lower()}")


def format_variant_line(name: str, decode: list[float], prefill: list[float]) -> str:
    """User format: `variantname  decode=X.XXtok/s(n=N)  prefill=Y.YYtok/s(n=N)` (medians)."""
    d = f"{statistics.median(decode):.2f}" if decode else "nan"
    p = f"{statistics.median(prefill):.2f}" if prefill else "nan"
    return f"{name}  decode={d}tok/s(n={len(decode)})  prefill={p}tok/s(n={len(prefill)})"


def check_proof_gate(proof_path: str, prov: dict, families: list[Family],
                     workloads: list[str]) -> list[str]:
    """Timing is refused unless a LIVE diagnostic summary for the SAME exe sha256
    and the SAME model inputs proved every requested family/workload."""
    probs: list[str] = []
    try:
        s = json.loads(Path(proof_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"cannot read proof summary {proof_path!r}: {exc}"]
    if s.get("mode") != "diagnostic" or s.get("dry_run") is not False:
        probs.append("proof summary is not a live diagnostic run")
    if not s.get("complete"):
        probs.append("proof run did not complete")
    pp = s.get("provenance", {})
    if not prov.get("exe_sha256") or pp.get("exe_sha256") != prov.get("exe_sha256"):
        probs.append("proof exe sha256 differs from timing exe")
    if pp.get("model_inputs") != prov.get("model_inputs"):
        probs.append("proof model inputs differ from timing inputs")
    if pp.get("model_input_identity") != prov.get("model_input_identity"):
        probs.append("proof model input identity (size/mtime) differs")
    if pp.get("prompts") != prov.get("prompts"):
        probs.append("proof prompts differ")
    verdicts = {(v.get("family"), v.get("workload")): v for v in s.get("pairs", [])}
    for fam in families:
        for wl in workloads:
            v = verdicts.get((fam.name, wl))
            if v is None:
                probs.append(f"no proof for {fam.name}/{wl}")
            elif v.get("proof") is not True:
                probs.append(f"proof failed for {fam.name}/{wl}")
    return probs


def _write_json(out_dir: Path | None, name: str, obj: dict) -> None:
    if out_dir is not None:
        (out_dir / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run(argv: list[str] | None = None) -> int:
    from tier_a import _config as TA  # noqa: PLC0415  (paths only; no tokenizer import)

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES), action="append",
                    help="family to run (repeatable); default: both, each independent")
    ap.add_argument("--workload", choices=WORKLOADS, action="append",
                    help="workload (repeatable); default: both")
    ap.add_argument("--mode", choices=("diagnostic", "timing"), required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="offline: build the plan and launch ZERO engines")
    ap.add_argument("--out", default="", metavar="DIR",
                    help="artifact root; must be new (or empty) for a live run")
    ap.add_argument("--proof", default="", metavar="SUMMARY_JSON",
                    help="timing only: summary.json of the passed live diagnostic run")
    ap.add_argument("--exe", default="/data/llm/Strata-tier-b/build-tier-b/strata")
    ap.add_argument("--pack", default=str(TA.PACK_DIR))
    ap.add_argument("--native", default=str(TA.NATIVE_GGUF))
    ap.add_argument("--ple-gguf", default=str(TA.PLE_SHARD))
    ap.add_argument("--expert-profile", default=str(TA.EXPERT_PROFILE))
    ap.add_argument("--mtp", default=str(TA.MTP_DIR))
    ap.add_argument("--per-arm-timeout-s", type=float, default=Bounds.per_arm_timeout_s)
    ap.add_argument("--session-cap-s", type=float, default=Bounds.session_cap_s)
    ap.add_argument("--recorded-pairs", type=int, default=Bounds.recorded_pairs)
    args = ap.parse_args(argv)

    families = [FAMILIES[f] for f in (args.family or sorted(FAMILIES))]
    workloads = args.workload or list(WORKLOADS)
    for fam in families:
        fam.validate()
    bounds = Bounds(per_arm_timeout_s=args.per_arm_timeout_s,
                    session_cap_s=args.session_cap_s,
                    recorded_pairs=args.recorded_pairs)
    bounds.validate()

    out_dir = Path(args.out).resolve() if args.out else None
    if not args.dry_run:
        probs = validate_live_inputs(args)
        if out_dir is None:
            probs.append("--out is required for a live run")
        elif out_dir.exists() and any(out_dir.iterdir()):
            probs.append(f"--out {out_dir} exists and is not empty (artifacts are never reused)")
        if args.mode == "timing" and not args.proof:
            probs.append("timing requires --proof (passed live diagnostic summary.json)")
        if probs:
            for p in probs:
                print(f"REFUSED: {p}", flush=True)
            return 2
    # Dry-run without --out: plan against a placeholder root that is never created.
    plan_root = out_dir or Path(tempfile.gettempdir()) / "tier_b_dry_run_unused"
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    base_flags = fixed_flags(args.pack, args.native, args.ple_gguf,
                             args.expert_profile, args.mtp)
    prompts = load_prompts(args.dry_run)
    if not args.dry_run:   # dry-run ids are synthetic; live ids must match counts + hashes
        assert_prompts_expected(prompts, workloads)
    base_env = dict(os.environ)
    controlled = ambient_strata_vars(base_env)
    prov = write_provenance(out_dir, args, bounds, families, workloads, prompts, base_env)

    if args.mode == "timing":
        if args.dry_run and not args.proof:
            print("proof gate: SKIPPED (dry-run; a live timing run requires --proof)", flush=True)
        else:
            probs = check_proof_gate(args.proof, prov, families, workloads)
            if probs:
                for p in probs:
                    print(f"REFUSED (proof gate): {p}", flush=True)
                return 2
            print(f"proof gate: PASS ({args.proof})", flush=True)

    print(f"Tier-B run_pairs  mode={args.mode}  {'[DRY RUN] ' if args.dry_run else ''}"
          f"families={[f.name for f in families]}  workloads={workloads}", flush=True)
    print("Contracts: " + ", ".join(
        f"{f.name}={'bitwise' if f.contract == 'bitwise' else f'rel<= {f.rel_tol:g}'}"
        for f in families), flush=True)
    print(f"Bounds: per_arm_timeout={bounds.per_arm_timeout_s}s session_cap={bounds.session_cap_s}s "
          f"max_windows={bounds.max_windows} max_export={bounds.max_export_bytes} "
          f"recorded_pairs={bounds.recorded_pairs} min_matched_windows={bounds.min_matched_windows}", flush=True)

    pairs = (plan_diagnostic(families, workloads, base_flags, plan_root)
             if args.mode == "diagnostic"
             else plan_timing(families, workloads, bounds, base_flags))

    clock = SessionClock(bounds)
    launches = 0
    pair_records: list[dict] = []
    samples: dict[tuple[str, str, str], dict[str, list]] = {}
    failures: list[str] = []
    complete = True

    with tempfile.TemporaryDirectory(prefix="tier_b_pairs_") as tmp_dir:
        for pair in pairs:
            ids = prompts[pair.workload]
            max_new = bounds.max_new_for(pair.workload)
            results: dict[str, ArmResult] = {}
            for spec in pair.launch_specs():
                timeout_s = clock.arm_timeout()
                if timeout_s is None:
                    print(f"SESSION CAP: {clock.elapsed():.0f}s of {bounds.session_cap_s:.0f}s used; "
                          f"refusing {spec.label} (run INCOMPLETE)", flush=True)
                    complete = False
                    break
                if spec.export_dir is not None and not args.dry_run:
                    if os.path.exists(spec.export_dir):
                        print(f"REFUSED: export dir already exists: {spec.export_dir}", flush=True)
                        return 2
                    # The exporter mkdir()s only the leaf: its parent must exist.
                    Path(spec.export_dir).parent.mkdir(parents=True, exist_ok=True)
                print(f"{'PLAN ' if args.dry_run else 'RUN  '}{spec.label}  "
                      f"({len(ids)} prompt ids, max_new={max_new}, timeout={timeout_s:.0f}s)",
                      flush=True)
                if not args.dry_run:
                    launches += 1
                res = run_arm(
                    exe=args.exe, flags=spec.flags, token_ids=ids, max_new=max_new,
                    env_overrides=spec.env_overrides, controlled_vars=controlled,
                    timeout_s=timeout_s, dry_run=args.dry_run, tmp_dir=tmp_dir,
                )
                _persist(None if args.dry_run else out_dir, spec, res, base_env, timeout_s)
                results[spec.arm] = res
                if not args.dry_run and res.error is not None:
                    print(f"FAIL-FAST {spec.label}: {res.error}", flush=True)
                    failures.append(f"{spec.label}: {res.error}")
                    complete = False
                    break
            if not complete:
                break
            if args.dry_run:
                continue

            if args.mode == "diagnostic":
                v = evaluate_diagnostic_pair(pair.family, pair.workload, results["A"], results["B"],
                                             pair.arm_a.export_dir or "", pair.arm_b.export_dir or "",
                                             bounds)
                pair_records.append(v)
                print(f"PAIR {pair.family.name}/{pair.workload}: proof={v['proof']} "
                      f"full_proof={v['full_proof']} windows={v['n_windows']} "
                      f"- {'; '.join(v['reasons'])}", flush=True)
                if not v["proof"]:
                    failures.append(f"{pair.family.name}/{pair.workload}: proof failed")
            else:
                ta = timing_sample(results["A"], len(ids))
                tb = timing_sample(results["B"], len(ids))
                same = results["A"].output_ids == results["B"].output_ids
                rec = {"family": pair.family.name, "workload": pair.workload,
                       "recorded": pair.recorded, "round_idx": pair.round_idx,
                       "order": "".join(pair.order), "outputs_match": same,
                       "A": ta._asdict(), "B": tb._asdict()}
                pair_records.append(rec)
                if not (ta.ok and tb.ok):
                    failures.append(f"{pair.family.name}/{pair.workload} r{pair.round_idx}: "
                                    f"invalid timing A={ta.reason} B={tb.reason}")
                    complete = False
                    break
                if not same:
                    failures.append(f"{pair.family.name}/{pair.workload} r{pair.round_idx}: "
                                    "A/B output IDs differ in a timing pair")
                if pair.recorded:
                    for arm, ts in (("A", ta), ("B", tb)):
                        s = samples.setdefault((pair.family.name, pair.workload, arm),
                                               {"decode": [], "prefill": []})
                        s["decode"].append(ts.decode_tok_s)
                        if len(ids) > 1:
                            s["prefill"].append(ts.prefill_tok_s)

    if args.dry_run:
        print(f"dry-run complete: planned {sum(len(p.launch_specs()) for p in pairs)} arms "
              f"in {len(pairs)} pairs, launched {launches} engines", flush=True)
        return 0

    summary: dict = {"mode": args.mode, "dry_run": False, "complete": complete and not failures,
                     "elapsed_s": clock.elapsed(), "launches": launches,
                     "provenance": prov, "pairs": pair_records, "failures": failures}
    if args.mode == "timing":
        stats: dict = {}
        lines: list[str] = []
        for fam in families:
            for wl in workloads:
                sa = samples.get((fam.name, wl, "A"), {"decode": [], "prefill": []})
                sb = samples.get((fam.name, wl, "B"), {"decode": [], "prefill": []})
                stats[f"{fam.name}/{wl}"] = {
                    "decode": summarise_pairs(sa["decode"], sb["decode"]),
                    "prefill": summarise_pairs(sa["prefill"], sb["prefill"]),
                }
                lines.append(f"[{wl}]")
                lines.append(format_variant_line(variant_name(fam, "A"), sa["decode"], sa["prefill"]))
                lines.append(format_variant_line(variant_name(fam, "B"), sb["decode"], sb["prefill"]))
        summary["paired_stats"] = stats
        summary["report_lines"] = lines
        print("\n".join(lines), flush=True)
    _write_json(out_dir, "summary.json", summary)
    ok = summary["complete"]
    print(f"{'PASS' if ok else 'FAIL'}: {args.mode} {launches} launches in {clock.elapsed():.0f}s"
          + ("" if ok else f" - {failures or ['incomplete']}"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())

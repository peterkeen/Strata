"""tools/tier_a/engine_cli.py - run one CLI arm (direct binary, no subcommand) and parse its output.

The engine binary is invoked as:

    /path/to/strata --pack DIR --tokens-file TMPFILE --max-new N [other flags]

stdout contains:
    prompt  : <id> <id> ...
    output  : <id> <id> ...
    decode                   N tokens in X.Y ms  ->  Z.ZZ tok/s
    prefill                  N tokens in X.Y ms  ->  Z.ZZ tok/s  (time to first token W.W ms)
    speculation              N rounds of S, drafts accepted A of T (R.RRR), P.PP tokens per round
    ... (--stats adds more lines to stderr)

stderr is captured raw (startup messages, expert-tier diagnostics, --stats output).

The module is pure Python (stdlib only; no strata_tokenizer / regex import).
It does NOT import _config so the offline tests can run without model paths.

DEPENDENCY NOTE
───────────────
strata_tokenizer.py (used by tokenize_prompt.py) requires the `regex` package,
which is not in the system Python on paseo-nibbler.  Before running live arms
or calling load_prompts_encoded() set:

    export PYTHONPATH=/home/pete/.cache/uv/archive-v0/wicKyd1x7CYO0Nj0/lib/python3.11/site-packages

This module itself has no such dependency; offline tests pass with bare Python.
"""
from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import NamedTuple


# ── output record ─────────────────────────────────────────────────────────────

class ArmResult(NamedTuple):
    prompt_ids:    list[int]   # ids as parsed from 'prompt  :' line
    output_ids:    list[int]   # ids as parsed from 'output  :' line
    output_hash:   str         # sha256[:16] of space-joined output ids
    n_output:      int         # len(output_ids)
    decode_ms:     float       # from 'decode' line
    decode_tok_s:  float
    prefill_ms:    float       # 0.0 if single-token prompt (no prefill line)
    prefill_tok_s: float
    ttft_ms:       float       # time-to-first-token; 0.0 if absent
    draft_accepted: int        # from 'speculation' line
    draft_offered:  int
    draft_rate:    float       # accepted / offered, or 0.0
    tokens_per_round: float
    stdout_raw:    str
    stderr_raw:    str
    wall_s:        float       # subprocess wall time
    exit_code:     int
    error:         str | None  # None = clean; else short description
    argv:          list[str]   # full command as launched (tokens-file path included)


# ── stdout parser ─────────────────────────────────────────────────────────────

_RE_PROMPT  = re.compile(r"^prompt\s+:((?:\s+-?\d+)+)", re.MULTILINE)
_RE_OUTPUT  = re.compile(r"^output\s+:((?:\s+-?\d+)*)", re.MULTILINE)
_RE_DECODE  = re.compile(
    r"^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.MULTILINE)
_RE_PREFILL = re.compile(
    r"^prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s"
    r".*?time to first token ([\d.]+) ms", re.MULTILINE)
_RE_SPEC    = re.compile(
    r"^speculation\s+(\d+) rounds of \d+, drafts accepted (\d+) of (\d+)"
    r" \(([\d.]+)\), ([\d.]+) tokens per round", re.MULTILINE)


def parse_stdout(text: str) -> dict:
    """Parse the engine's stdout.  Returns a dict with all extracted fields.
    Missing sections produce zero/empty defaults so callers always get every key."""
    out: dict = {}

    m = _RE_PROMPT.search(text)
    out["prompt_ids"] = [int(x) for x in m.group(1).split()] if m else []

    m = _RE_OUTPUT.search(text)
    raw_ids = [int(x) for x in m.group(1).split()] if m else []
    out["output_ids"] = raw_ids
    out["n_output"]   = len(raw_ids)
    id_str = " ".join(str(i) for i in raw_ids)
    out["output_hash"] = hashlib.sha256(id_str.encode()).hexdigest()[:16]

    m = _RE_DECODE.search(text)
    if m:
        out["decode_header_n"] = int(m.group(1))    # count from 'decode N tokens' line
        out["decode_ms"]       = float(m.group(2))
        out["decode_tok_s"]    = float(m.group(3))
    else:
        out["decode_header_n"] = -1  # no decode line present
        out["decode_ms"] = out["decode_tok_s"] = 0.0

    m = _RE_PREFILL.search(text)
    if m:
        out["prefill_ms"]    = float(m.group(2))
        out["prefill_tok_s"] = float(m.group(3))
        out["ttft_ms"]       = float(m.group(4))
    else:
        out["prefill_ms"] = out["prefill_tok_s"] = out["ttft_ms"] = 0.0

    m = _RE_SPEC.search(text)
    if m:
        out["draft_accepted"]   = int(m.group(2))
        out["draft_offered"]    = int(m.group(3))
        out["draft_rate"]       = float(m.group(4))
        out["tokens_per_round"] = float(m.group(5))
    else:
        out["draft_accepted"] = out["draft_offered"] = 0
        out["draft_rate"] = out["tokens_per_round"] = 0.0

    return out


# ── environment builder ────────────────────────────────────────────────────────

def build_env(base_env: dict[str, str],
              controlled_vars: tuple[str, ...],
              arm_overrides: dict[str, str]) -> dict[str, str]:
    """Inherit base_env, strip controlled_vars, apply arm_overrides.
    Prevents ambient STRATA_* settings from leaking between arms."""
    env = {k: v for k, v in base_env.items() if k not in controlled_vars}
    env.update(arm_overrides)
    return env


# ── helpers ────────────────────────────────────────────────────────────────────

def math_finite(x) -> bool:
    import math
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _write_ids_tmpfile(ids: list[int], tmp_dir: str) -> tuple[int, str]:
    """Create a temp file via mkstemp, write space-separated ids, return (fd, path).
    The caller is responsible for closing fd and eventually unlinking path.
    mkstemp avoids the race window that tempfile.mktemp() leaves between
    name-generation and file creation."""
    fd, path = tempfile.mkstemp(suffix=".ids", dir=tmp_dir)
    try:
        os.write(fd, " ".join(str(i) for i in ids).encode("ascii"))
    except Exception:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return fd, path


def _sentinel(
    token_ids: list[int],
    error: str,
    argv: list[str],
    stdout_raw: str = "",
    stderr_raw: str = "",
    exit_code: int = 0,
    wall_s: float = 0.0,
) -> ArmResult:
    return ArmResult(
        prompt_ids=list(token_ids), output_ids=[], output_hash=error[:16],
        n_output=0, decode_ms=0.0, decode_tok_s=0.0,
        prefill_ms=0.0, prefill_tok_s=0.0, ttft_ms=0.0,
        draft_accepted=0, draft_offered=0, draft_rate=0.0, tokens_per_round=0.0,
        stdout_raw=stdout_raw, stderr_raw=stderr_raw,
        wall_s=wall_s, exit_code=exit_code, error=error, argv=argv,
    )


# ── single arm runner ──────────────────────────────────────────────────────────

def run_arm(
    exe:             str | Path,
    flags:           list[str],
    token_ids:       list[int],
    max_new:         int,
    env_overrides:   dict[str, str],
    controlled_vars: tuple[str, ...],
    timeout_s:       float = 300.0,
    dry_run:         bool  = False,
    tmp_dir:         str | None = None,
) -> ArmResult:
    """Launch one engine process, capture stdout/stderr, return ArmResult.

    The --tokens-file temp file is created with mkstemp (no race window),
    closed before the child process opens it, and unlinked in a finally block
    that runs on every path: normal exit, timeout, launch failure, and dry_run.
    The full argv (including the tokens-file path) is recorded in ArmResult.argv
    so artifact logs can reproduce or audit the invocation.
    """
    tmp_dir = tmp_dir or tempfile.gettempdir()

    # Create temp file, close fd, child will open it read-only.
    fd, ids_path = _write_ids_tmpfile(token_ids, tmp_dir)
    os.close(fd)

    cmd: list[str] = [
        str(exe),
        "--tokens-file", ids_path,
        "--max-new", str(max_new),
        *flags,
    ]

    env = build_env(dict(os.environ), controlled_vars, env_overrides)

    try:
        if dry_run:
            return _sentinel(token_ids, "dry_run", cmd)

        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            return _sentinel(token_ids, f"launch_failed: {exc}", cmd,
                             wall_s=time.monotonic() - t0)

        try:
            stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout_b, stderr_b = proc.communicate()
            wall_s = time.monotonic() - t0
            return _sentinel(
                token_ids, f"timeout after {timeout_s:.0f}s", cmd,
                stdout_raw=stdout_b.decode("utf-8", "replace"),
                stderr_raw=stderr_b.decode("utf-8", "replace"),
                exit_code=-1, wall_s=wall_s,
            )

        wall_s    = time.monotonic() - t0
        stdout_tx = stdout_b.decode("utf-8", "replace")
        stderr_tx = stderr_b.decode("utf-8", "replace")
        exit_code = proc.returncode

        error: str | None = None
        if exit_code != 0:
            error = f"exit_code={exit_code}"
        elif not stdout_tx.strip():
            error = "empty_stdout"

        parsed = parse_stdout(stdout_tx)

        if error is None:
            n = parsed["n_output"]
            dec = parsed["decode_ms"]
            if n == 0:
                error = "zero_output_tokens"
            elif not math_finite(dec) or dec <= 0:
                error = f"nonfinite_or_zero decode_ms={dec}"
            elif not math_finite(parsed["decode_tok_s"]):
                error = f"nonfinite decode_tok_s={parsed['decode_tok_s']}"

        # Decode header count must match output_ids count.
        if error is None:
            hdr_n = parsed.get("decode_header_n", -1)
            if hdr_n >= 0 and hdr_n != parsed["n_output"]:
                error = (
                    f"decode_count_mismatch: "
                    f"header={hdr_n} output_ids={parsed['n_output']}"
                )

        # Prompt echo must match supplied token_ids.
        if error is None:
            parsed_prompt = parsed.get("prompt_ids", [])
            if not parsed_prompt and token_ids:
                error = "missing_prompt_echo"
            elif parsed_prompt != list(token_ids):
                error = (
                    f"prompt_mismatch: supplied {len(token_ids)} ids, "
                    f"engine echoed {len(parsed_prompt)}"
                )

        return ArmResult(
            prompt_ids    = parsed.get("prompt_ids", list(token_ids)),
            output_ids    = parsed["output_ids"],
            output_hash   = parsed["output_hash"],
            n_output      = parsed["n_output"],
            decode_ms     = parsed["decode_ms"],
            decode_tok_s  = parsed["decode_tok_s"],
            prefill_ms    = parsed["prefill_ms"],
            prefill_tok_s = parsed["prefill_tok_s"],
            ttft_ms       = parsed["ttft_ms"],
            draft_accepted    = parsed["draft_accepted"],
            draft_offered     = parsed["draft_offered"],
            draft_rate        = parsed["draft_rate"],
            tokens_per_round  = parsed["tokens_per_round"],
            stdout_raw    = stdout_tx,
            stderr_raw    = stderr_tx,
            wall_s        = wall_s,
            exit_code     = exit_code,
            error         = error,
            argv          = list(cmd),
        )
    finally:
        try:
            os.unlink(ids_path)
        except OSError:
            pass

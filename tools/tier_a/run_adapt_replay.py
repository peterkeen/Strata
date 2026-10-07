"""tools/tier_a/run_adapt_replay.py - Adapt-replay harness using --spec-follow.

Runs 9 arms (adapt-every off/every1/every2 × 3 reps, same rotation as run_matrix.py
G5) with --spec-follow pointing at a canonical reference output_ids snapshot.  This
forces every arm to emit the same token sequence as the reference, so decode_tok_s is
comparable across adapt variants independent of output divergence.

The `follow N differ of M` diagnostic counts positions where the natural argmax
differed from the forced continuation — an argmax disagreement count, not a quality
or generation signal.

CAVEAT
──────
--spec-follow is forced continuation: output ids come from the reference, NOT from
free generation.  decode_tok_s figures are still cold-graph (see METHODOLOGY_CAVEAT
in run_matrix.py).  Equal output sequence does NOT guarantee equal verify-window T
distribution or graph-capture cost.  follow_differ is an argmax disagreement count,
not a direct expert routing metric.

DEPENDENCY NOTE
───────────────
Live runs need `regex` via:
    export PYTHONPATH=/home/pete/.cache/uv/archive-v0/wicKyd1x7CYO0Nj0/lib/python3.11/site-packages

Usage:
    python tools/tier_a/run_adapt_replay.py --reference PATH --out DIR [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from tier_a._config import (   # noqa: E402
    CANDIDATE_EXE, FIXED_FLAGS, FUSED_ENV, STRATA_CONTROLLED_VARS,
)
from tier_a.engine_cli import ArmResult, run_arm   # noqa: E402
from tier_a.run_matrix import (   # noqa: E402
    _base_flags, _persist_arm, _prompt_hash, load_prompts_encoded,
    TIMEOUT_SHORT,
)

# ── constants ─────────────────────────────────────────────────────────────────

ADAPT_REPLAY_CAVEAT = (
    "spec-follow forced continuation: output ids are fixed to the canonical reference "
    "snapshot; decode_tok_s reflects processing cost under the controlled sequence, "
    "NOT free generation or quality. "
    "Cold-graph triage — each process compiles CUDA graphs on first use; "
    "equal output sequence does NOT guarantee equal verify-window T distribution "
    "or graph-capture cost across arms. "
    "follow_differ counts positions where the natural argmax differed from the "
    "forced continuation; this is an argmax disagreement count, "
    "not a direct expert routing metric."
)

TIMEOUT_REPLAY = TIMEOUT_SHORT   # story prompt; 300s is ample with forced continuation

_RE_FOLLOW = re.compile(
    r"^follow\s+(\d+) of (\d+) emitted tokens differ from the argmax",
    re.MULTILINE,
)

# Synthetic reference used in --dry-run (221 ids, all non-negative, >= 200).
_SYNTHETIC_REFERENCE: list[int] = list(range(1000, 1221))


# ── reference helpers ─────────────────────────────────────────────────────────

def _load_reference(path: str) -> list[int]:
    """Read and validate reference output_ids.txt (once; immutable after return).

    Validates:
      - file exists
      - every token is a non-negative integer
      - at least 200 ids (minimum useful replay length)

    Raises FileNotFoundError or ValueError on any problem.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reference file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"reference file is empty: {path}")
    ids: list[int] = []
    for tok in text.split():
        try:
            v = int(tok)
        except ValueError:
            raise ValueError(
                f"reference file contains non-integer token: {tok!r} in {path}"
            )
        if v < 0:
            raise ValueError(
                f"reference file contains negative id {v} in {path}"
            )
        ids.append(v)
    if len(ids) < 200:
        raise ValueError(
            f"reference file has only {len(ids)} ids; "
            f"need >=200 for a valid replay ({path})"
        )
    return ids


def _reference_hash(ids: list[int]) -> str:
    return hashlib.sha256(" ".join(str(i) for i in ids).encode()).hexdigest()[:16]


# ── follow diagnostic parser ──────────────────────────────────────────────────

def _parse_follow(stdout: str) -> tuple[int, int]:
    """Parse 'follow  N of M emitted tokens differ from the argmax' from stdout.

    Returns (differ, emitted); both -1 if the line is absent (e.g. dry_run,
    error exit, or engine built without --spec-follow active).
    """
    m = _RE_FOLLOW.search(stdout)
    if m:
        return int(m.group(1)), int(m.group(2))
    return -1, -1


# ── arm builder ───────────────────────────────────────────────────────────────

def _build_replay_arms(reference_path: str, reference_n: int) -> list[dict]:
    """Build 9 arms: adapt-every off/every1/every2 × 3 reps, G5 rotation.

    Rotation is identical to run_matrix.py G5: start index advances by rep,
    so each rep begins on a different variant.  Adds --spec-follow to force
    the reference continuation on every arm.
    """
    adapt_variants: list[tuple[str, list[str]]] = [
        ("off",    ["--adapt-every", "0"]),
        ("every1", ["--adapt-every", "1", "--adapt-swaps", "80"]),
        ("every2", ["--adapt-every", "2", "--adapt-swaps", "80"]),
    ]
    arms: list[dict] = []
    for rep in range(3):
        start = rep % len(adapt_variants)
        rotated = adapt_variants[start:] + adapt_variants[:start]
        for ae_variant, ae_flags in rotated:
            flags = _base_flags("auto", ae_flags) + ["--spec-follow", reference_path]
            arms.append(dict(
                label      = f"g5r_rep{rep}_{ae_variant}",
                exe        = str(CANDIDATE_EXE),
                flags      = flags,
                env        = dict(FUSED_ENV),
                prompt_key = "story",
                max_new    = reference_n,
                timeout    = TIMEOUT_REPLAY,
                variant    = ae_variant,
                note       = (
                    f"adapt-every {ae_variant}; spec-follow forced continuation; "
                    "decode_tok_s comparable across variants (same output sequence)"
                ),
            ))
    return arms


# ── main ──────────────────────────────────────────────────────────────────────

def run(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--reference", required=True, metavar="PATH",
        help="reference output_ids.txt from a prior off-adapt arm (321 ids including EOS)",
    )
    ap.add_argument(
        "--out", required=True, metavar="DIR",
        help="write per-arm artifacts and summary JSON here",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="offline: skip GPU and file reads; use synthetic reference ids",
    )
    ap.add_argument(
        "--fail-fast", action="store_true",
        help=(
            "abort on the first arm error or output mismatch; "
            "per-arm artifact is written before the abort"
        ),
    )
    ap.add_argument(
        "--timeout-scale", type=float, default=1.0,
        help="multiply all timeouts by this factor",
    )
    args = ap.parse_args(argv)

    dry_run = args.dry_run
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── reference ─────────────────────────────────────────────────────────────
    # source_reference_path: caller-supplied path, recorded for provenance only.
    # canonical_path: snapshot written into out_dir; --spec-follow uses this path.
    source_reference_path = str(Path(args.reference).resolve())

    if dry_run:
        reference_ids: list[int] = list(_SYNTHETIC_REFERENCE)
        reference_hash = _reference_hash(reference_ids)
        print(
            f"[DRY RUN] synthetic reference: {len(reference_ids)} ids  "
            f"hash={reference_hash}",
            flush=True,
        )
    else:
        print(f"Loading reference: {args.reference}", flush=True)
        reference_ids = _load_reference(args.reference)
        reference_hash = _reference_hash(reference_ids)
        print(f"  {len(reference_ids)} ids  hash={reference_hash}", flush=True)

    # Write canonical snapshot atomically.  All arms point --spec-follow at this
    # file, not at the original.  If the original is modified or deleted after
    # this point the arms are unaffected.
    snap     = out_dir / "reference_ids.txt"
    snap_tmp = out_dir / "reference_ids.txt.tmp"
    snap_tmp.write_text(" ".join(str(i) for i in reference_ids), encoding="utf-8")
    snap_tmp.rename(snap)
    canonical_path = str(snap.resolve())

    # Reference metadata — written atomically.
    _meta = {
        "source_reference_path": source_reference_path,
        "reference_path":        canonical_path,
        "reference_hash":        reference_hash,
        "reference_n":           len(reference_ids),
        "dry_run":               dry_run,
        "caveat":                ADAPT_REPLAY_CAVEAT,
    }
    _meta_tmp = out_dir / "reference_meta.json.tmp"
    _meta_tmp.write_text(json.dumps(_meta, indent=2), encoding="utf-8")
    _meta_tmp.rename(out_dir / "reference_meta.json")

    print(f"\nCaveat: {ADAPT_REPLAY_CAVEAT}", flush=True)

    # ── prompts ────────────────────────────────────────────────────────────────
    print("\nLoading prompts...", flush=True)
    prompts = load_prompts_encoded(dry_run)
    token_ids = prompts["story"]
    prompt_hash_val = _prompt_hash(token_ids)
    print(f"  story: {len(token_ids)} tokens  prompt_hash={prompt_hash_val}", flush=True)

    # ── arms ──────────────────────────────────────────────────────────────────
    arms = _build_replay_arms(canonical_path, len(reference_ids))
    print(
        f"\n{len(arms)} arms  "
        f"(adapt-every off/every1/every2 × 3 reps, G5 rotation)\n",
        flush=True,
    )

    all_records: list[dict] = []
    t_total = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="adapt_replay_") as tmp_dir:
        for arm in arms:
            timeout = arm["timeout"] * args.timeout_scale

            print(
                f"RUN   {arm['label']}  "
                f"({len(token_ids)} prompt tokens, max_new={arm['max_new']}, "
                f"timeout={timeout:.0f}s)",
                flush=True,
            )

            result: ArmResult = run_arm(
                exe             = arm["exe"],
                flags           = arm["flags"],
                token_ids       = token_ids,
                max_new         = arm["max_new"],
                env_overrides   = arm["env"],
                controlled_vars = STRATA_CONTROLLED_VARS,
                timeout_s       = timeout,
                dry_run         = dry_run,
                tmp_dir         = tmp_dir,
            )

            # ── follow diagnostic ──────────────────────────────────────────────
            follow_differ, follow_emitted = _parse_follow(result.stdout_raw)
            if follow_differ >= 0:
                print(
                    f"  follow: {follow_differ} of {follow_emitted} "
                    "positions differ from argmax",
                    flush=True,
                )

            # ── eligibility ────────────────────────────────────────────────────
            performance_eligible = True
            performance_ineligible_reason: str | None = None

            if result.error not in (None, "dry_run"):
                performance_eligible = False
                performance_ineligible_reason = f"error: {result.error}"
            elif not dry_run and result.error is None:
                if result.output_ids != reference_ids:
                    performance_eligible = False
                    performance_ineligible_reason = (
                        f"output_mismatch: got {result.n_output} ids, "
                        f"expected {len(reference_ids)}"
                    )
                    print(
                        f"  OUTPUT MISMATCH: got {result.n_output} ids, "
                        f"expected {len(reference_ids)} — "
                        "performance_eligible=False; raw artifact preserved",
                        flush=True,
                    )

            status = (
                f"ERROR({result.error})"   if result.error not in (None, "dry_run") else
                "MISMATCH"                  if not performance_eligible else
                "ok"
            )
            print(
                f"  status={status}  exit={result.exit_code}  wall={result.wall_s:.1f}s\n"
                f"  output={result.n_output} tokens  hash={result.output_hash}\n"
                f"  decode={result.decode_ms:.1f}ms  {result.decode_tok_s:.2f}tok/s  "
                f"prefill={result.prefill_ms:.1f}ms  {result.prefill_tok_s:.1f}tok/s",
                flush=True,
            )

            record: dict = {
                "label":            arm["label"],
                "variant":          arm["variant"],
                "exe":              arm["exe"],
                "prompt_key":       arm["prompt_key"],
                "prompt_hash":      prompt_hash_val,
                "n_prompt":         len(token_ids),
                "max_new":          arm["max_new"],
                "note":             arm["note"],
                "reference_hash":         reference_hash,
                "reference_path":         canonical_path,
                "source_reference_path":  source_reference_path,
                "reference_n":            len(reference_ids),
                "follow_differ":    follow_differ,
                "follow_emitted":   follow_emitted,
                "output_ids_match_reference": (
                    result.output_ids == reference_ids
                    if not dry_run and result.error is None
                    else None   # not checked in dry_run or when an error occurred
                ),
                "caveat":           ADAPT_REPLAY_CAVEAT,
                "performance_eligible":          performance_eligible,
                "performance_ineligible_reason": performance_ineligible_reason,
                "n_output":         result.n_output,
                "output_hash":      result.output_hash,
                "decode_ms":        result.decode_ms,
                "decode_tok_s":     result.decode_tok_s,
                "prefill_ms":       result.prefill_ms,
                "prefill_tok_s":    result.prefill_tok_s,
                "ttft_ms":          result.ttft_ms,
                "draft_accepted":   result.draft_accepted,
                "draft_offered":    result.draft_offered,
                "draft_rate":       result.draft_rate,
                "tokens_per_round": result.tokens_per_round,
                "wall_s":           result.wall_s,
                "exit_code":        result.exit_code,
                "error":            result.error,
                "effective_env":    {
                    k: arm["env"].get(k, "(unset)") for k in STRATA_CONTROLLED_VARS
                },
            }
            all_records.append(record)

            # Persist immediately — interrupt loses nothing already measured.
            _persist_arm(out_dir, arm, result, record, token_ids)

            # ── fail-fast ──────────────────────────────────────────────────────
            if (
                args.fail_fast
                and not performance_eligible
                and result.error not in ("dry_run",)
            ):
                print(
                    f"\nFAIL-FAST triggered by {arm['label']}: "
                    f"{performance_ineligible_reason}",
                    flush=True,
                )
                return 1

    elapsed = time.monotonic() - t_total

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Total wall time: {elapsed:.0f}s", flush=True)

    real_errors = [
        r for r in all_records if r.get("error") not in (None, "dry_run")
    ]
    mismatches = [
        r for r in all_records if r.get("output_ids_match_reference") is False
    ]

    if real_errors:
        print(f"\nERRORS ({len(real_errors)}):", flush=True)
        for r in real_errors:
            print(f"  {r['label']}: {r['error']}", flush=True)

    if mismatches:
        print(f"\nOUTPUT MISMATCHES ({len(mismatches)}):", flush=True)
        for r in mismatches:
            print(
                f"  {r['label']}: {r['performance_ineligible_reason']}",
                flush=True,
            )

    # Median follow_differ per adapt variant (argmax disagreement count).
    by_variant: dict[str, list[int]] = {}
    for r in all_records:
        if r["follow_differ"] >= 0:
            by_variant.setdefault(r["variant"], []).append(r["follow_differ"])
    if by_variant:
        print("\nMedian follow_differ per variant:", flush=True)
        for v in sorted(by_variant):
            vals = by_variant[v]
            print(
                f"  {v}: {statistics.median(vals):.0f} of "
                f"{len(reference_ids)} positions (n={len(vals)})",
                flush=True,
            )

    print(f"\nCaveat: {ADAPT_REPLAY_CAVEAT}", flush=True)

    # ── write summary JSON ────────────────────────────────────────────────────
    payload = {
        "caveat":                ADAPT_REPLAY_CAVEAT,
        "reference_hash":        reference_hash,
        "reference_path":        canonical_path,
        "source_reference_path": source_reference_path,
        "reference_n":           len(reference_ids),
        "elapsed_s":             round(elapsed, 1),
        "results":               all_records,
    }
    json_path = out_dir / "adapt_replay_results.json"
    tmp_path  = out_dir / "adapt_replay_results.json.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.rename(json_path)
    print(f"\nWrote {json_path}", flush=True)

    return 1 if (real_errors or mismatches) else 0


if __name__ == "__main__":
    sys.exit(run())

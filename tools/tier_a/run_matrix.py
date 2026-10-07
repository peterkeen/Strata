"""tools/tier_a/run_matrix.py - Tier-A benchmark harness (CLI fresh-process mode).

METHODOLOGY CAVEAT (read before interpreting results)
──────────────────────────────────────────────────────
Each arm is a fresh OS process.  The engine compiles CUDA graphs for the decode
loop the first time each verify-window size T is exercised in that process.
Graph compilation appears as CUDA JIT work inside the measured decode wall clock.

Therefore ALL decode_tok_s figures in this harness are cold-graph triage numbers,
NOT steady-state throughput.  Specifically:

  • Even matched token budgets do NOT guarantee equal graph-capture cost between
    arms: capture cost depends on the actual T distribution encountered, the
    order in which windows are hit, and the text being generated.  Two arms with
    the same max_new budget may encounter different T values if their output
    diverges, and prompt hash equality must be verified before comparing decode
    numbers across arms.
  • prefill_tok_s is unaffected (the prefill path uses a separate batched kernel
    that is not captured into the token graph).
  • For steady-state decode throughput a persistent --serve harness with
    per-arm warmup generations is required.  That is out of scope for this task;
    the parent decides whether the cold-graph triage numbers are sufficient.

Comparisons this harness IS designed to support:
  • baseline EXE vs candidate EXE (same prompt, same flags, alternating order)
  • --prefill auto vs --prefill 4096 (prefill_tok_s is the signal)
  • --adapt-every 0 vs 1 vs 2 (decode_tok_s; order rotated across reps)
  • STRATA_GR_DOWN_MAX4 off vs on (one pair, recorded as likely_noop)

Comparisons labelled NOT_APPLICABLE:
  • STRATA_IQ256_GATHER: AMD 9900X routes IQ3_S through native_gu_rows
    (AVX-512), not cpu_gather_fast.  Gather knob has no effect on this host.

DEPENDENCY NOTE
───────────────
live runs (not --dry-run) need `regex` via:
    export PYTHONPATH=/home/pete/.cache/uv/archive-v0/wicKyd1x7CYO0Nj0/lib/python3.11/site-packages

Usage:
  python tools/tier_a/run_matrix.py --dry-run          # offline; no GPU needed
  python tools/tier_a/run_matrix.py --smoke --out DIR  # 1 rep, 32/64 tokens
  python tools/tier_a/run_matrix.py --out DIR          # full matrix (~20 min)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from tier_a._config import (   # noqa: E402
    ADAPT_OFF, BASELINE_EXE, CANDIDATE_EXE, ELIGIBILITY, FIXED_FLAGS,
    FUSED_ENV, STRATA_CONTROLLED_VARS, TOKENIZER_DIR,
)
from tier_a.engine_cli import ArmResult, run_arm   # noqa: E402
from tier_a.prompts import LONG_DOC_Q, SHORT_CODE, STORY   # noqa: E402
# tokenize_prompt is imported lazily inside load_prompts_encoded() so that
# importing this module (e.g. in tests) does not require `regex`.

# ── constants ─────────────────────────────────────────────────────────────────

METHODOLOGY_CAVEAT = (
    "CLI fresh-process cold-graph triage: every decode_tok_s includes CUDA graph "
    "compilation for the first occurrence of each verify-window size T in that "
    "process.  NOT steady-state throughput.  prefill_tok_s unaffected.  "
    "Matched token budgets do not guarantee equal graph-capture cost; "
    "verify prompt_hash equality before comparing decode numbers across arms."
)

TIMEOUT_LONG  = 600.0   # long-prompt arms
TIMEOUT_SHORT = 180.0   # short/story arms

# Full-run minimum output threshold: arms with max_new >= MIN_FULL_BUDGET_TOKENS
# that return fewer than MIN_OUTPUT_FULL tokens are flagged NOT as performance
# data but as short-output events (EOS early / prompt mismatch).
MIN_FULL_BUDGET_TOKENS = 128
MIN_OUTPUT_FULL        = 200


# ── arm definitions ───────────────────────────────────────────────────────────

def _base_flags(prefill: str = "auto", adapt_flags: list[str] | None = None) -> list[str]:
    flags = list(FIXED_FLAGS)
    flags += adapt_flags if adapt_flags is not None else list(ADAPT_OFF)
    flags += ["--prefill", prefill]
    return flags


def _arm(
    label:      str,
    exe:        str | Path,
    flags:      list[str],
    env:        dict[str, str],
    prompt_key: str,
    max_new:    int,
    timeout:    float,
    variant:    str = "",   # explicit variant tag used by summarise_group
    note:       str = "",
) -> dict:
    return dict(label=label, exe=str(exe), flags=flags, env=env,
                prompt_key=prompt_key, max_new=max_new, timeout=timeout,
                variant=variant, note=note)


def build_matrix(smoke: bool = False) -> list[dict]:
    """Return the arm list.  Order within each group is counterbalanced."""
    arms: list[dict] = []
    base_env = dict(FUSED_ENV)

    max_short = 32  if smoke else 256
    max_long  = 64  if smoke else 512
    reps      = 1   if smoke else 3

    # ── Groups 1–3: baseline vs candidate ────────────────────────────────────
    # Counterbalanced order per rep: rep0 → B,C  rep1 → C,B  rep2 → B,C
    # This rotates assignment so rep0 and rep2 start with baseline and rep1
    # starts with candidate, reducing systematic first-process advantages.
    bc_orders = [
        [("baseline", BASELINE_EXE), ("candidate", CANDIDATE_EXE)],  # rep0
        [("candidate", CANDIDATE_EXE), ("baseline", BASELINE_EXE)],  # rep1
        [("baseline", BASELINE_EXE), ("candidate", CANDIDATE_EXE)],  # rep2
    ]

    for g, (prompt_key, max_new, timeout, note_suffix) in enumerate([
        ("short_code", max_short, TIMEOUT_SHORT, "short code; cold-graph triage"),
        ("long_doc",   max_long,  TIMEOUT_LONG,  "8K+ doc prompt; prefill_tok_s primary signal"),
        ("story",      max_long,  TIMEOUT_SHORT, "story; target ≥200 output tokens"),
    ], start=1):
        for rep in range(reps):
            order = bc_orders[rep % len(bc_orders)]
            for who, exe in order:
                arms.append(_arm(
                    label      = f"g{g}_rep{rep}_{who}",
                    exe        = exe,
                    flags      = _base_flags("auto"),
                    env        = base_env,
                    prompt_key = prompt_key,
                    max_new    = max_new,
                    timeout    = timeout,
                    variant    = who,
                    note       = note_suffix,
                ))

    if smoke:
        return arms

    # ── Group 4: prefill auto vs 4096, candidate, long doc ───────────────────
    pf_orders = [
        [("auto", "auto"), ("fixed4096", "4096")],
        [("fixed4096", "4096"), ("auto", "auto")],
        [("auto", "auto"), ("fixed4096", "4096")],
    ]
    for rep in range(reps):
        for pf_variant, pf_val in pf_orders[rep % len(pf_orders)]:
            arms.append(_arm(
                label      = f"g4_rep{rep}_{pf_variant}",
                exe        = CANDIDATE_EXE,
                flags      = _base_flags(pf_val),
                env        = base_env,
                prompt_key = "long_doc",
                max_new    = max_long,
                timeout    = TIMEOUT_LONG,
                variant    = pf_variant,
                note       = f"prefill {pf_val}; prefill_tok_s is the primary signal",
            ))

    # ── Group 5: adapt-every 0/1/2, candidate, story ─────────────────────────
    # Rotate the three variants across reps to avoid a fixed ordering bias.
    adapt_variants = [
        ("off",    ["--adapt-every", "0"]),
        ("every1", ["--adapt-every", "1", "--adapt-swaps", "80"]),
        ("every2", ["--adapt-every", "2", "--adapt-swaps", "80"]),
    ]
    for rep in range(reps):
        # Rotate start index by rep so each rep starts on a different variant.
        rotated = adapt_variants[rep % len(adapt_variants):] + \
                  adapt_variants[:rep % len(adapt_variants)]
        for ae_variant, ae_flags in rotated:
            arms.append(_arm(
                label      = f"g5_rep{rep}_{ae_variant}",
                exe        = CANDIDATE_EXE,
                flags      = _base_flags("auto", ae_flags),
                env        = base_env,
                prompt_key = "story",
                max_new    = max_long,
                timeout    = TIMEOUT_SHORT,
                variant    = ae_variant,
                note       = f"adapt-every {ae_variant}; decode_tok_s signal",
            ))

    # ── Group 6: GR_DOWN_MAX4 off/on, one pair ───────────────────────────────
    for grd_variant, grd_val in (("off", None), ("on", "1")):
        env_grd = dict(base_env)
        if grd_val is not None:
            env_grd["STRATA_GR_DOWN_MAX4"] = grd_val
        arms.append(_arm(
            label      = f"g6_{grd_variant}",
            exe        = CANDIDATE_EXE,
            flags      = _base_flags("auto"),
            env        = env_grd,
            prompt_key = "story",
            max_new    = max_long,
            timeout    = TIMEOUT_SHORT,
            variant    = grd_variant,
            note       = ELIGIBILITY["GR_DOWN_MAX4"],
        ))

    # ── Group 7: IQ256_GATHER — not_applicable, skipped ──────────────────────
    arms.append(_arm(
        label      = "g7_iq256_gather_not_applicable",
        exe        = CANDIDATE_EXE,
        flags      = _base_flags("auto"),
        env        = base_env,
        prompt_key = "short_code",
        max_new    = max_short,
        timeout    = TIMEOUT_SHORT,
        variant    = "not_applicable",
        note       = ELIGIBILITY["IQ256_GATHER"] + "; SKIPPED (not_applicable)",
    ))

    return arms


# ── prompt encoding ───────────────────────────────────────────────────────────

def load_prompts_encoded(dry_run: bool) -> dict[str, list[int]]:
    """Return {key: token_id_list}.  Dry-run returns synthetic ids."""
    if dry_run:
        return {
            "short_code": list(range(1, 60)),
            "long_doc":   list(range(1, 9000)),   # 8999 synthetic ids > 8192
            "story":      list(range(1, 120)),
        }
    from tier_a.tokenize_prompt import encode_chat, load_tokenizer  # noqa: PLC0415
    tok = load_tokenizer(TOKENIZER_DIR)
    prompts = {
        "short_code": encode_chat(tok, SHORT_CODE),
        "long_doc":   encode_chat(tok, LONG_DOC_Q),
        "story":      encode_chat(tok, STORY),
    }
    n_long = len(prompts["long_doc"])
    if n_long < 8192:
        raise RuntimeError(
            f"long_doc encodes to only {n_long} tokens; need ≥8192.  "
            "Expand the corpus in prompts.py."
        )
    return prompts


# ── artifact helpers ──────────────────────────────────────────────────────────

def _arm_artifact_dir(out_dir: Path, label: str) -> Path:
    """Return and create the per-arm artifact subdirectory."""
    d = out_dir / "arms" / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def _persist_arm(out_dir: Path | None, arm: dict, result: ArmResult,
                 record: dict, prompt_ids: list[int]) -> None:
    """Write per-arm raw files immediately after each run.
    Files are written atomically (write to .tmp then rename) so an interrupted
    matrix run leaves complete records for all arms that finished."""
    if out_dir is None:
        return
    d = _arm_artifact_dir(out_dir, arm["label"])

    def _write(name: str, content: str) -> None:
        tmp = d / (name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.rename(d / name)

    _write("stdout.txt", result.stdout_raw)
    _write("stderr.txt", result.stderr_raw)
    _write("argv.txt",   "\n".join(result.argv))
    _write("prompt_ids.txt", " ".join(str(i) for i in prompt_ids))
    _write("output_ids.txt",
           " ".join(str(i) for i in result.output_ids) if result.output_ids else "")
    _write("record.json", json.dumps(record, indent=2))


# ── result formatting ─────────────────────────────────────────────────────────

def _fmt_result(r: ArmResult, label: str) -> str:
    status = f"ERROR({r.error})" if r.error else "ok"
    return (
        f"  {label}\n"
        f"    status={status}  exit={r.exit_code}  wall={r.wall_s:.1f}s\n"
        f"    output={r.n_output} tokens  hash={r.output_hash}\n"
        f"    decode={r.decode_ms:.1f}ms  {r.decode_tok_s:.2f}tok/s  "
        f"prefill={r.prefill_ms:.1f}ms  {r.prefill_tok_s:.1f}tok/s  "
        f"ttft={r.ttft_ms:.1f}ms\n"
        f"    drafts accepted={r.draft_accepted}/{r.draft_offered}  "
        f"rate={r.draft_rate:.3f}  tok/round={r.tokens_per_round:.2f}\n"
    )


def summarise_group(records: list[dict]) -> str:
    """Median decode_tok_s and prefill_tok_s per arm['variant'] tag.

    Only records with performance_eligible=True (or absent, for backwards
    compat) are included.  Error records and short-output records (which
    have performance_eligible=False) are excluded from the medians but
    their raw data is preserved in the JSON.
    """
    by_variant_dec:  dict[str, list[float]] = {}
    by_variant_pre:  dict[str, list[float]] = {}
    for rec in records:
        # Exclude anything explicitly flagged ineligible (short-output, errors)
        if rec.get("performance_eligible") is False:
            continue
        # Legacy compat: also skip records with real errors that pre-date the field
        if rec.get("error") and rec["error"] not in (None, "dry_run"):
            continue
        v = rec.get("variant", "unknown")
        tok_s = rec.get("decode_tok_s", 0.0)
        if tok_s and tok_s > 0:
            by_variant_dec.setdefault(v, []).append(tok_s)
        pre_s = rec.get("prefill_tok_s", 0.0)
        if pre_s and pre_s > 0:
            by_variant_pre.setdefault(v, []).append(pre_s)
    if not by_variant_dec and not by_variant_pre:
        return "  (no eligible results)"
    all_variants = sorted(set(by_variant_dec) | set(by_variant_pre))
    parts = []
    for v in all_variants:
        dec_vals = by_variant_dec.get(v, [])
        pre_vals = by_variant_pre.get(v, [])
        dec_str = f"decode={statistics.median(dec_vals):.2f}tok/s(n={len(dec_vals)})" if dec_vals else "decode=N/A"
        pre_str = f"prefill={statistics.median(pre_vals):.2f}tok/s(n={len(pre_vals)})" if pre_vals else "prefill=N/A"
        parts.append(f"{v}  {dec_str}  {pre_str}")
    return "  " + "  |  ".join(parts)


# ── main ──────────────────────────────────────────────────────────────────────

def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="offline: skip GPU, synthetic ids, validate harness flow")
    ap.add_argument("--smoke", action="store_true",
                    help="1 rep, 32/64 tokens, groups 1-3 only")
    ap.add_argument("--out", default="", metavar="DIR",
                    help="write per-arm artifacts + summary JSON here; "
                         "each arm is persisted immediately after it finishes")
    ap.add_argument("--timeout-scale", type=float, default=1.0,
                    help="multiply all timeouts by this factor")
    ap.add_argument("--fail-fast", action="store_true",
                    help="abort the matrix on the first arm error (non-zero exit, "
                         "timeout, launch failure, parse error); short-output does "
                         "not trigger fail-fast")
    args = ap.parse_args(argv)

    dry_run = args.dry_run
    smoke   = args.smoke or dry_run

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tier-A benchmark  "
          f"{'[DRY RUN] ' if dry_run else ''}{'[SMOKE] ' if smoke else ''}",
          flush=True)
    print(f"Caveat: {METHODOLOGY_CAVEAT}", flush=True)

    print("\nLoading prompts...", flush=True)
    prompts = load_prompts_encoded(dry_run)
    for k, ids in prompts.items():
        print(f"  {k}: {len(ids)} tokens", flush=True)
    print(flush=True)

    matrix = build_matrix(smoke=smoke)

    all_records: list[dict] = []
    t_total = time.monotonic()
    cur_group = ""

    with tempfile.TemporaryDirectory(prefix="tier_a_") as tmp_dir:
        for arm in matrix:
            # ── skip not_applicable ──────────────────────────────────────────
            if arm["variant"] == "not_applicable":
                print(f"SKIP  {arm['label']}", flush=True)
                rec = {"label": arm["label"], "variant": arm["variant"],
                       "skipped": True, "note": arm["note"]}
                all_records.append(rec)
                _persist_arm(out_dir, arm,
                             ArmResult([], [], "skip", 0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0, 0, 0.0, 0.0, "", "", 0.0, 0,
                                       "skipped", []),
                             rec, [])
                continue

            # ── group header ─────────────────────────────────────────────────
            grp_key = arm["label"].split("_rep")[0]
            if grp_key != cur_group:
                cur_group = grp_key
                print(f"\n── {grp_key} ──", flush=True)

            token_ids = prompts[arm["prompt_key"]]
            timeout   = arm["timeout"] * args.timeout_scale

            print(f"RUN   {arm['label']}  "
                  f"({len(token_ids)} prompt tokens, max_new={arm['max_new']}, "
                  f"timeout={timeout:.0f}s)",
                  flush=True)

            result: ArmResult = run_arm(
                exe              = arm["exe"],
                flags            = arm["flags"],
                token_ids        = token_ids,
                max_new          = arm["max_new"],
                env_overrides    = arm["env"],
                controlled_vars  = STRATA_CONTROLLED_VARS,
                timeout_s        = timeout,
                dry_run          = dry_run,
                tmp_dir          = tmp_dir,
            )

            print(_fmt_result(result, arm["label"]), end="", flush=True)

            # ── eligibility: short-output and errors ─────────────────────────
            performance_eligible        = True
            performance_ineligible_reason: str | None = None

            if result.error not in (None, "dry_run"):
                performance_eligible = False
                performance_ineligible_reason = f"error: {result.error}"
            elif (
                arm["max_new"] >= MIN_FULL_BUDGET_TOKENS
                and result.n_output < MIN_OUTPUT_FULL
            ):
                performance_eligible = False
                performance_ineligible_reason = (
                    f"short_output: n_output={result.n_output} < "
                    f"{MIN_OUTPUT_FULL}, budget={arm['max_new']}"
                )
                print(
                    f"      SHORT-OUTPUT: {result.n_output} tokens "
                    f"(budget={arm['max_new']}, threshold={MIN_OUTPUT_FULL}); "
                    f"EOS fired early — performance_eligible=False, "
                    f"raw evidence preserved",
                    flush=True,
                )

            prompt_hash = _prompt_hash(token_ids)

            record: dict = {
                "label":        arm["label"],
                "variant":      arm["variant"],
                "exe":          arm["exe"],
                "prompt_key":   arm["prompt_key"],
                "prompt_hash":  prompt_hash,
                "n_prompt":     len(token_ids),
                "max_new":      arm["max_new"],
                "note":         arm["note"],
                "methodology":  METHODOLOGY_CAVEAT,
                "performance_eligible":         performance_eligible,
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

            # Persist immediately — interrupt won't lose finished arms.
            _persist_arm(out_dir, arm, result, record, token_ids)

            # ── fail-fast ────────────────────────────────────────────────────
            if args.fail_fast and result.error not in (None, "dry_run"):
                print(
                    f"\nFAIL-FAST triggered by {arm['label']}: {result.error}",
                    flush=True,
                )
                return 1

    elapsed = time.monotonic() - t_total

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Total wall time: {elapsed:.0f}s", flush=True)

    real_errors = [r for r in all_records
                   if not r.get("skipped")
                   and r.get("error") not in (None, "dry_run")]
    if real_errors:
        print(f"\nERRORS ({len(real_errors)}):", flush=True)
        for r in real_errors:
            print(f"  {r['label']}: {r['error']}", flush=True)

    # Group summaries keyed by group prefix (before first _rep or _ split)
    groups: dict[str, list[dict]] = {}
    for r in all_records:
        if r.get("skipped"):
            continue
        gk = r["label"].split("_rep")[0]
        groups.setdefault(gk, []).append(r)

    print("\nGroup summaries (median decode tok/s per variant):", flush=True)
    for gk, recs in groups.items():
        print(f"  {gk}:", flush=True)
        print(summarise_group(recs), flush=True)

    print(f"\nCaveat: {METHODOLOGY_CAVEAT}", flush=True)

    # ── write summary JSON ────────────────────────────────────────────────────
    payload = {
        "methodology": METHODOLOGY_CAVEAT,
        "eligibility": ELIGIBILITY,
        "elapsed_s":   round(elapsed, 1),
        "results":     all_records,
    }

    if out_dir:
        json_path = out_dir / "tier_a_results.json"
        tmp_path  = out_dir / "tier_a_results.json.tmp"
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.rename(json_path)
        print(f"\nWrote {json_path}", flush=True)
    else:
        blob = json.dumps(payload, indent=2)
        print("\n--- JSON ---", flush=True)
        print(blob[:4000], flush=True)
        if len(blob) > 4000:
            print("... (truncated; use --out DIR for full output)", flush=True)

    return 1 if real_errors else 0


def _prompt_hash(ids: list[int]) -> str:
    s = " ".join(str(i) for i in ids)
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:16]


if __name__ == "__main__":
    sys.exit(run())

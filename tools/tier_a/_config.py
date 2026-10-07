"""tools/tier_a/_config.py - model paths and fixed launch parameters for the Tier-A benchmark harness.

All paths are consulted only when --dry-run is not active and when the harness actually launches an engine
process; the offline unit tests never import them.  The tokenizer is loaded once per harness run via
strata_tokenizer.Tokenizer (same approach as tools/calibrate.py), so the production .venv's lack of
`tokenizers` is not a problem.
"""
from __future__ import annotations

from pathlib import Path

# ── model artefacts ───────────────────────────────────────────────────────────

PACK_DIR        = Path("/data/llm/Strata-data/packs/iq3_s")
NATIVE_GGUF     = Path("/data/llm/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S"
                        "/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
PLE_SHARD       = Path("/data/llm/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S"
                        "/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")
EXPERT_PROFILE  = Path("/data/llm/Strata/data/expert-profile.bin")
MTP_DIR         = Path("/data/llm/Strata-data/mtp/rt")
BASELINE_EXE    = Path("/data/llm/Strata-tests/tier-a-20261007/production.strata")
CANDIDATE_EXE   = Path("/data/llm/Strata-tier-a/build-tier-a/strata")

# Pack tokenizer-sharp (contains vocab.json / merges.txt / token_type.json).
TOKENIZER_DIR   = PACK_DIR / "tokenizer-sharp"

# ── fixed flags shared by both arms ──────────────────────────────────────────
# No batch size / no --serve: we use direct CLI per the smoke confirmation.
# --stop-eos so natural turn endings are respected; output count is checked post-run.
# STRATA_IQ_MT_MIN=1 fixes CPU row dispatch count for matched A/B determinism.
# STRATA_NO_MULTI_GR is left unset (engine default) for both arms.

FIXED_FLAGS: list[str] = [
    # Pack directory: required; without it the engine defaults to the Q2_0 pack
    # (or fails), giving completely wrong kernel paths and timing.
    "--pack",             str(PACK_DIR),
    # Native shard paths
    "--native",           str(NATIVE_GGUF),
    "--ple-gguf",         str(PLE_SHARD),
    "--expert-profile",   str(EXPERT_PROFILE),
    "--mtp",              str(MTP_DIR),
    # Expert cache: 2000 slots matches the production setup that produced
    # 2582 resident slots in the upstream smoke run; omitting this defaults
    # to cache=0 which disables the VRAM expert path entirely.
    "--expert-cache",     "2000",
    # PCIe fraction: 0 pins the PCIe-vs-CPU split at zero so no expert
    # compute migrates over PCIe during benchmarking; eliminates a source
    # of arm-to-arm variance from adaptive PCIe scheduling.
    "--pcie-frac",        "0",
    # Context / KV
    "--max-context",      "262144",
    "--kv-resident",      "32768",
    "--kv",               "int8",
    # Speculation
    "--spec",             "4",
    "--spec-min-p",       "0.5",
    "--vram-reserve-mib", "2048",
    # --stats: writes per-round detail to stderr; required so the artifact
    # logs contain cache-hit rates and slot counts for post-run audit.
    "--stats",
    # Stop at EOS so full-run arms produce natural output lengths.
    "--stop-eos",
]

# Fused prompt env for both arms.
FUSED_ENV: dict[str, str] = {"STRATA_PF_FUSED": "1", "STRATA_IQ_MT_MIN": "1"}

# Environment variables that may affect kernel selection; strip inherited values
# for clean A/B arms and re-inject only the explicit per-arm setting.
STRATA_CONTROLLED_VARS: tuple[str, ...] = (
    "STRATA_GR_DOWN_MAX4",
    "STRATA_IQ256_GATHER",
    "STRATA_NO_AVXVNNI",
    "STRATA_IQ_MT_MIN",
    "STRATA_PF_FUSED",
    "STRATA_PF_FUSED_NATIVE",
    "STRATA_NO_MULTI_GR",
)

# Adaptation off: --adapt-every 0 confirmed working in smoke; removes tracking
# overhead and prevents cache movement between arms.
ADAPT_OFF: list[str] = ["--adapt-every", "0"]

# ── eligibility notes ─────────────────────────────────────────────────────────
#
# IQ256_GATHER: AMD 9900X has AVX-512.  native_expert.cpp routes IQ3_S to
# native_gu_rows (AVX-512) before the AVX-2 cpu_gather_fast path is reached.
# cpu_gather_fast itself only probes Intel AVX-VNNI, not AVX-512.  The gather
# knob therefore has no effect on this host; labelled not_applicable, not run.
#
# GR_DOWN_MAX4: fused_gr.cu lines 884-910 — the max4 branch guards ct<=4 &&
# gr_down_max4().  With STRATA_NO_MULTI_GR unset (default), exact-T specialised
# kernels (ct==1..6) are chosen first for T=1..6, covering all single-token and
# small-multi-token decode.  The max4 generic path is unreachable in normal
# decode.  One pair recorded as a no-op confirmation.

ELIGIBILITY: dict[str, str] = {
    "IQ256_GATHER": (
        "not_applicable: AMD 9900X has AVX-512; native_expert.cpp routes IQ3_S to "
        "native_gu_rows (AVX-512) before cpu_gather_fast; cpu_gather_fast probes "
        "Intel AVX-VNNI only — gather path not active on this host"
    ),
    "GR_DOWN_MAX4": (
        "likely_noop: exact-T staged kernels (ct==1..6) selected first when "
        "STRATA_NO_MULTI_GR unset (default); max4 generic branch unreachable for "
        "T<=6 single/multi-token decode; one pair run to confirm"
    ),
}

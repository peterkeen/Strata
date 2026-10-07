"""tools/tier_a/tokenize_prompt.py - load the pack tokenizer and encode prompts.

Uses strata_tokenizer.Tokenizer exactly as tools/calibrate.py does:
  - vocab from vocab.json (list index = token id)
  - merges from merges.txt
  - types from token_type.json
  - regex dependency is tools/strata_tokenizer.py's only non-stdlib import

No `tokenizers` package is needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402


def load_tokenizer(tokenizer_dir: Path) -> ST.Tokenizer:
    """Load the pack tokenizer from `tokenizer_dir` (vocab.json / merges.txt / token_type.json)."""
    vocab: dict[str, int] = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
    toks: list[str | None] = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    merges_text = (tokenizer_dir / "merges.txt").read_text(encoding="utf-8")
    merges = merges_text.split("\n")
    token_types: list[int] = json.loads((tokenizer_dir / "token_type.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(toks, merges, token_types)  # type: ignore[arg-type]


def encode_chat(tok: ST.Tokenizer, text: str) -> list[int]:
    """Encode a pre-formatted chat string (already contains im_start/im_end) with parse_special=True."""
    return tok.encode(text, parse_special=True)

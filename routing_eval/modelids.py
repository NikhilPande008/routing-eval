"""ALLOWED_MODELS parsing + Fireworks model-ID normalization.

Shared by conformance.py (the scoring shell) and policy.py (the routing
policy) -- both need this and neither should import the other for it (see
taskio.py for the same problem/pattern with task-dict helpers).

Confirmed live 2026-07-09 against the real Fireworks endpoint: this
project's ALLOWED_MODELS env var gives bare model names (e.g. "minimax-m3"),
but Fireworks' /chat/completions API 404s on a bare name -- it requires the
full "accounts/fireworks/models/<name>" path. Both minimax-m3 and
kimi-k2p7-code confirmed 404 unprefixed, 200 prefixed. Not a hypothetical:
this is what broke the first real call in Step 1's dry run.
"""
from __future__ import annotations

from typing import List

_FIREWORKS_MODEL_PREFIX = "accounts/fireworks/models/"


def normalize_model_id(model: str) -> str:
    """Idempotent: a string that already looks like a full path (contains
    '/') is left untouched, so this is safe to apply more than once or to an
    already-correct value."""
    return model if "/" in model else _FIREWORKS_MODEL_PREFIX + model


def split_models(allowed_models: str) -> List[str]:
    """ALLOWED_MODELS is comma/whitespace-separated; used by `score` (first
    model only), Step 2's bake-off, and Step 3's policy fallback -- all of
    them get normalized IDs from this single parse point."""
    models = [m.strip() for m in allowed_models.replace(",", " ").split() if m.strip()]
    if not models:
        raise ValueError("ALLOWED_MODELS is empty -- no model to call")
    return [normalize_model_id(m) for m in models]

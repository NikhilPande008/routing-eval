"""Pluggable accuracy scorers. Each returns a float in [0,1] (1.0 = correct).

The competition's exact metric is revealed at kickoff, but it will almost
certainly fall into one of these buckets. Add the real one here and everything
downstream (records, frontier, operating point) works unchanged.
"""
from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List

_WS = re.compile(r"\s+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUM = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def _norm(s: Any, strip_punct: bool = False, drop_articles: bool = False) -> str:
    s = str(s).strip().casefold()
    if strip_punct:
        s = s.translate(str.maketrans("", "", string.punctuation))
    if drop_articles:
        s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def score_exact(pred: Any, gold: Any, strip_punct: bool = False,
                drop_articles: bool = False, **_: Any) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    p = _norm(pred, strip_punct, drop_articles)
    return 1.0 if any(p == _norm(g, strip_punct, drop_articles) for g in golds) else 0.0


def score_numeric(pred: Any, gold: Any, rel_tol: float = 1e-6,
                  abs_tol: float = 1e-9, **_: Any) -> float:
    nums = _NUM.findall(str(pred))
    if not nums:
        return 0.0
    val = float(nums[-1].replace(",", "").replace("$", ""))   # answer is usually last
    try:
        return 1.0 if math.isclose(val, float(gold), rel_tol=rel_tol, abs_tol=abs_tol) else 0.0
    except (TypeError, ValueError):
        return 0.0


def score_multiple_choice(pred: Any, gold: Any, allowed: List[str] = None, **_: Any) -> float:
    if not allowed:
        return score_exact(pred, gold)
    low = _norm(pred)
    last = None                                    # take the LAST mentioned label
    for lab in allowed:
        for m in re.finditer(r"\b" + re.escape(lab.casefold()) + r"\b", low):
            if last is None or m.start() > last[0]:
                last = (m.start(), lab)
    if last is None:
        return 0.0
    return 1.0 if last[1].casefold() == str(gold).casefold() else 0.0


def _tokens(s: Any) -> List[str]:
    return _norm(s, strip_punct=True, drop_articles=True).split()


def score_token_f1(pred: Any, gold: Any, **_: Any) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    pt = _tokens(pred)
    best = 0.0
    for g in golds:
        gt = _tokens(g)
        if not pt and not gt:
            best = max(best, 1.0)
            continue
        if not pt or not gt:
            continue
        overlap = sum((Counter(pt) & Counter(gt)).values())
        if overlap == 0:
            continue
        prec, rec = overlap / len(pt), overlap / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def score_json_match(pred: Any, gold: Any, mode: str = "exact", **_: Any) -> float:
    txt = re.sub(r"^```(?:json)?|```$", "", str(pred).strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        return 0.0
    goldobj = json.loads(gold) if isinstance(gold, str) else gold
    if mode == "exact":
        return 1.0 if obj == goldobj else 0.0
    if mode == "subset" and isinstance(obj, dict) and isinstance(goldobj, dict):
        return 1.0 if all(obj.get(k) == v for k, v in goldobj.items()) else 0.0
    return 0.0


def score_code_tests(pred: Any, gold: Any, timeout: float = 5.0, **_: Any) -> float:
    """Run model code + provided tests (gold holds the test harness) in a subprocess.

    WARNING: executes model-generated code. Only run inside a sandbox (the
    scoring env, or a throwaway container) -- never on your dev host unguarded.
    Provided as a documented interface; NOT exercised by the stand-in datasets.
    """
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(str(pred) + "\n\n" + str(gold))
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return 1.0 if r.returncode == 0 else 0.0
    except Exception:
        return 0.0
    finally:
        os.unlink(path)


SCORERS: Dict[str, Callable[..., float]] = {
    "exact": score_exact,
    "numeric": score_numeric,
    "multiple_choice": score_multiple_choice,
    "token_f1": score_token_f1,
    "json_match": score_json_match,
    "code_tests": score_code_tests,
}


def score(pred: Any, item) -> float:
    """Score a prediction against an Item using that item's declared scorer."""
    fn = SCORERS[item.scorer]
    opts = dict(item.scorer_opts)
    if item.scorer == "multiple_choice" and item.allowed and "allowed" not in opts:
        opts["allowed"] = item.allowed
    return float(fn(pred, item.gold, **opts))

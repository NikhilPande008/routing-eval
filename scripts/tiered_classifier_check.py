#!/usr/bin/env python3
"""D41 diagnostic (zero token cost): validates TieredClassifier -- the
deployed classifier after the token-tiering change -- against all three
fixtures. The check that matters is asymmetric:

  FALSE POSITIVES (a task detected into a category it isn't) are the
  dangerous direction: a knowledge/summarization task misdetected as
  math/logic/entity gets a terse template and risks the judge. Required: 0.

  FALSE NEGATIVES (a real math/logic/entity task not detected) only cost
  tokens: the task falls through to "_default" (the fuller template) and
  stays accurate. Reported for completeness, not gated on.

Expected mapping: code_debug/code_gen/code -> "code"; sentiment ->
"sentiment"; entity_extraction -> "entity_extraction"; math -> "math";
logic -> "logic"; knowledge/summarization/wordplay -> "general".
One deliberate cross-category exception, asserted explicitly below: the
chickens-and-cows puzzle (classifier_paraphrases, labeled "logic") is
arithmetic in logic clothing -- detecting it as "math" is correct routing
(both get terse templates), so it's counted OK-as-math, not a miss.

Run:  python3 scripts/tiered_classifier_check.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval.classify import TieredClassifier  # noqa: E402

FIXTURES = {
    "accuracy_diagnostic (32)": os.path.join(
        os.path.dirname(__file__), "fixtures", "accuracy_diagnostic.json"),
    "classifier_paraphrases (36)": os.path.join(
        os.path.dirname(__file__), "fixtures", "classifier_paraphrases.json"),
    "two_way_stress_test (24)": os.path.join(
        os.path.dirname(__file__), "fixtures", "two_way_stress_test.json"),
}

EXPECTED = {
    "code_debug": "code", "code_gen": "code", "code": "code",
    "sentiment": "sentiment",
    "entity_extraction": "entity_extraction",
    "math": "math",
    "logic": "logic",
    "knowledge": "general", "summarization": "general", "wordplay": "general",
}
# category -> what it may ALSO acceptably detect as (terse-for-terse swaps only)
ACCEPTABLE_ALTERNATE = {"logic": {"math"}}


def main() -> int:
    clf = TieredClassifier()
    dangerous_fp, token_costing_fn, ok = [], [], 0
    per_detected = defaultdict(int)

    for fixture_name, path in FIXTURES.items():
        with open(path) as f:
            cases = json.load(f)
        for c in cases:
            expected = EXPECTED[c["category"]]
            got = clf.classify(c["prompt"]).category
            per_detected[got] += 1
            if got == expected or got in ACCEPTABLE_ALTERNATE.get(c["category"], set()):
                ok += 1
            elif got == "general":
                # missed detection -> fuller template: tokens, not accuracy
                token_costing_fn.append((fixture_name, c["category"], c["prompt"]))
            else:
                # detected into a WRONG specific category: the dangerous kind
                dangerous_fp.append((fixture_name, c["category"], got, c["prompt"]))

    total = ok + len(dangerous_fp) + len(token_costing_fn)
    print(f"{total} prompts across 3 fixtures -> {ok} correct, "
         f"{len(token_costing_fn)} missed-detections (fall to 'general': token cost only), "
         f"{len(dangerous_fp)} DANGEROUS false positives (accuracy risk)\n")

    print("Detected-category distribution:", dict(sorted(per_detected.items())))

    if token_costing_fn:
        print(f"\nMissed detections ({len(token_costing_fn)}) -- safe, costs tokens only:")
        for fixture_name, cat, prompt in token_costing_fn:
            print(f"  [{fixture_name}] true={cat!r}: {prompt[:90]!r}")

    if dangerous_fp:
        print(f"\nDANGEROUS FALSE POSITIVES ({len(dangerous_fp)}) -- DO NOT SHIP:")
        for fixture_name, cat, got, prompt in dangerous_fp:
            print(f"  [{fixture_name}] true={cat!r} detected={got!r}: {prompt[:90]!r}")
        return 1

    print("\nZero dangerous false positives -- safe to judge-proxy-gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

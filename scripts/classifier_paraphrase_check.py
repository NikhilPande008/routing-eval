#!/usr/bin/env python3
"""Diagnostic (not a test): checks whether classify.py's KeywordClassifier
generalizes beyond the exact 8 practice-task prompts it was calibrated
against (D28). Runs it against scripts/fixtures/classifier_paraphrases.json
-- hand-written paraphrases of each category that avoid the literal trigger
keyword where avoidable -- and reports per-category hit rate plus every
misclassification. Report only; does not fix anything (2026-07-09).

Run:  python3 scripts/classifier_paraphrase_check.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval.classify import KeywordClassifier  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "classifier_paraphrases.json")


def main() -> int:
    with open(FIXTURE) as f:
        cases = json.load(f)

    clf = KeywordClassifier()
    per_category = defaultdict(lambda: {"hits": 0, "total": 0})
    misclassifications = []

    for case in cases:
        expected = case["category"]
        result = clf.classify(case["prompt"])
        per_category[expected]["total"] += 1
        if result.category == expected:
            per_category[expected]["hits"] += 1
        else:
            misclassifications.append((expected, result.category, result.confidence,
                                       result.matched, case["prompt"]))

    print(f"{'category':<18}{'hit rate':>10}   detail")
    total_hits = total_n = 0
    for cat in sorted(per_category):
        h, n = per_category[cat]["hits"], per_category[cat]["total"]
        total_hits += h
        total_n += n
        print(f"{cat:<18}{h}/{n:<8}{h/n:>6.0%}")
    print(f"\nOVERALL: {total_hits}/{total_n} ({total_hits/total_n:.0%})")

    if misclassifications:
        print(f"\n{len(misclassifications)} misclassification(s):")
        for expected, got, confidence, matched, prompt in misclassifications:
            print(f"  expected={expected!r} got={got!r} confidence={confidence:.2f}")
            print(f"    prompt: {prompt[:90]!r}")
            print(f"    matched: {matched}")
    else:
        print("\nNo misclassifications.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""D37 diagnostic (not a test, zero token cost): measures false-positive and
false-negative rates for the two narrow detectors that replaced the 8-way
KeywordClassifier in the deployed path -- `is_code_task` and
`is_sentiment_task` (routing_eval/classify.py).

Runs both detectors against TWO fixtures:
  1. scripts/fixtures/accuracy_diagnostic.json -- the 32-task diagnostic
     (8 categories x 4 tasks), the same set D35/D36 used.
  2. scripts/fixtures/classifier_paraphrases.json -- 36 hand-written
     paraphrases that avoid each category's trigger keyword where avoidable
     (D29's exact fixture) -- the harder, keyword-avoiding test.

For each detector, a task is:
  - a positive (should fire) if its true category is "code_debug"/"code_gen"
    (for is_code_task) or "sentiment" (for is_sentiment_task).
  - a negative (should NOT fire) otherwise.
False positive  = detector fires on a negative.
False negative  = detector does NOT fire on a positive.

Run:  python3 scripts/two_way_detector_check.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval.classify import is_code_task, is_sentiment_task  # noqa: E402

FIXTURES = {
    "accuracy_diagnostic (32 tasks)": os.path.join(
        os.path.dirname(__file__), "fixtures", "accuracy_diagnostic.json"),
    "classifier_paraphrases (36 tasks)": os.path.join(
        os.path.dirname(__file__), "fixtures", "classifier_paraphrases.json"),
    "two_way_stress_test (24 tasks, held out during design)": os.path.join(
        os.path.dirname(__file__), "fixtures", "two_way_stress_test.json"),
}

CODE_CATEGORIES = {"code_debug", "code_gen", "code"}
SENTIMENT_CATEGORIES = {"sentiment"}


def load(path):
    with open(path) as f:
        return json.load(f)


def evaluate(name, tasks, detector, positive_categories):
    fp, fn, tp, tn = [], [], 0, 0
    for t in tasks:
        category = t["category"]
        prompt = t["prompt"]
        fired = detector(prompt)
        is_positive = category in positive_categories
        if is_positive and fired:
            tp += 1
        elif is_positive and not fired:
            fn.append((t.get("task_id", "?"), category, prompt))
        elif not is_positive and fired:
            fp.append((t.get("task_id", "?"), category, prompt))
        else:
            tn += 1
    n_pos = tp + len(fn)
    n_neg = tn + len(fp)
    print(f"  {name:<20} positives={n_pos:<3} negatives={n_neg:<3} "
         f"FN={len(fn)}/{n_pos} ({(len(fn)/n_pos if n_pos else 0):.0%})  "
         f"FP={len(fp)}/{n_neg} ({(len(fp)/n_neg if n_neg else 0):.0%})")
    return fp, fn


def main() -> int:
    all_fp = {"is_code_task": [], "is_sentiment_task": []}
    all_fn = {"is_code_task": [], "is_sentiment_task": []}

    for fixture_name, path in FIXTURES.items():
        tasks = load(path)
        print(f"\n=== {fixture_name} ===")
        fp, fn = evaluate("is_code_task", tasks, is_code_task, CODE_CATEGORIES)
        all_fp["is_code_task"] += [(fixture_name, *row) for row in fp]
        all_fn["is_code_task"] += [(fixture_name, *row) for row in fn]

        fp, fn = evaluate("is_sentiment_task", tasks, is_sentiment_task, SENTIMENT_CATEGORIES)
        all_fp["is_sentiment_task"] += [(fixture_name, *row) for row in fp]
        all_fn["is_sentiment_task"] += [(fixture_name, *row) for row in fn]

    print("\n" + "=" * 78)
    print("DETAIL: every false positive / false negative across both fixtures")
    print("=" * 78)
    for detector in ("is_code_task", "is_sentiment_task"):
        print(f"\n--- {detector} ---")
        if not all_fp[detector] and not all_fn[detector]:
            print("  none.")
            continue
        for fixture_name, task_id, category, prompt in all_fp[detector]:
            print(f"  FALSE POSITIVE [{fixture_name}] {task_id} (true category={category!r})")
            print(f"    prompt: {prompt[:100]!r}")
        for fixture_name, task_id, category, prompt in all_fn[detector]:
            print(f"  FALSE NEGATIVE [{fixture_name}] {task_id} (true category={category!r})")
            print(f"    prompt: {prompt[:100]!r}")

    total_fp = sum(len(v) for v in all_fp.values())
    total_fn = sum(len(v) for v in all_fn.values())
    print(f"\nTOTAL across both detectors, both fixtures: {total_fp} false positives, "
         f"{total_fn} false negatives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""D37 diagnostic: runs the FULL deployed score path -- classify (now
TwoWayClassifier) -> policy lookup -> prompt template -> real Fireworks call
-- over all 32 tasks in scripts/fixtures/accuracy_diagnostic.json, using the
checked-in default policy exactly as `routing-eval score` would.

This is the diagnostic D36 recommended and explicitly deferred: D35 measured
MODEL accuracy by bypassing the classifier (feeding each task its TRUE
category's template directly); D36 measured the OLD 8-way classifier's
misroute rate in isolation (zero token cost, no model call). Neither ran the
real path end-to-end. This script does: each task is handed to PolicyRouter
as a bare {task_id, prompt} dict -- exactly the real /input/tasks.json shape
(D24), with NO category field -- so the classifier genuinely has to guess,
exactly as it will at grading time.

Scoring: reports the raw declared-scorer score AND a "corrected" score that
applies the two scorer-artifact fixes D35 hand-verified (not new gold
values, just fairer grading of the SAME hand-verified gold):
  - sentiment: compare only the extracted label word, not full-sentence
    token_f1 (which tanks precision against a 1-word gold even when the
    label is exactly right, D35 finding #1).
  - "exact"-scored logic tasks: also accept the gold word/phrase appearing
    as a whole word inside a full-sentence answer (D35 finding #2 -- kimi's
    "Sam owns the cat." was marked wrong by strict exact-match against
    "Sam", but is correct content).
Numeric and code_tests scores are already reliable (D35) and left as-is.
token_f1 on summarization/entity_extraction remains a proxy only -- read the
raw answer yourself; no correction is applied because D35 found no scorer
bug there, just scorer looseness that needs a human eye.

Run:
  set -a && source .env && set +a
  export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
  python3 scripts/deployed_path_diagnostic.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval import scorers  # noqa: E402
from routing_eval.llm import OpenAICompatibleClient  # noqa: E402
from routing_eval.modelids import split_models  # noqa: E402
from routing_eval.policy import DEFAULT_POLICY_PATH, PolicyRouter, load_policy, resolve_entry  # noqa: E402
from routing_eval.schema import Item  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "accuracy_diagnostic.json")
RESULTS_OUT = "/tmp/deployed_path_diagnostic_results.json"
PROXY_ONLY_CATEGORIES = {"sentiment", "summarization", "entity_extraction"}
CODE_CATEGORIES = {"code_debug", "code_gen"}
EXPECTED_TEMPLATE = {
    "code_debug": "code_only", "code_gen": "code_only", "sentiment": "sentiment_with_justification",
}


def load_tasks():
    with open(FIXTURE) as f:
        return json.load(f)


def _label_only(answer: str) -> str:
    m = re.match(r"\s*([A-Za-z]+)", answer)
    return m.group(1).strip().casefold() if m else ""


def score_task(task, answer):
    item = Item(id=task["task_id"], task_type=task["category"], input=task["prompt"],
               gold=task["gold"], scorer=task["scorer"], scorer_opts=task.get("scorer_opts") or {})
    raw = scorers.score(answer, item)
    corrected = raw
    if task["category"] == "sentiment":
        gold_label = str(task["gold"]).strip().casefold()
        corrected = 1.0 if _label_only(answer) == gold_label else raw
    elif task["scorer"] == "exact":
        norm_gold = scorers._norm(task["gold"], strip_punct=True, drop_articles=True)
        norm_answer = scorers._norm(answer, strip_punct=True, drop_articles=True)
        if norm_gold and re.search(r"\b" + re.escape(norm_gold) + r"\b", norm_answer):
            corrected = max(raw, 1.0)
    return raw, corrected


def main() -> int:
    base_url = os.environ["FIREWORKS_BASE_URL"]
    api_key = os.environ["FIREWORKS_API_KEY"]
    allowed_models = split_models(os.environ["ALLOWED_MODELS"])
    client = OpenAICompatibleClient(base_url, api_key=api_key)
    policy = load_policy(DEFAULT_POLICY_PATH)

    tasks = load_tasks()
    router = PolicyRouter(policy=policy, remote_client=client, allowed_models=allowed_models)

    results = []
    misroutes = []
    for idx, task in enumerate(tasks):
        # Real /input/tasks.json shape (D24): task_id + prompt ONLY. No
        # category/gold field is handed to the router -- it must classify
        # blind, exactly like at grading time.
        bare_task = {"task_id": task["task_id"], "prompt": task["prompt"]}
        out = router.route_task(bare_task, idx)
        answer = out["answer"]

        detected_category = router.classifier.classify(task["prompt"]).category
        entry = resolve_entry(policy, detected_category)
        template_used = entry.prompt_template
        expected_template = EXPECTED_TEMPLATE.get(task["category"], "default")
        template_ok = template_used == expected_template
        if not template_ok:
            misroutes.append((task["task_id"], task["category"], detected_category, template_used,
                              expected_template))

        raw, corrected = score_task(task, answer)
        row = {"task_id": task["task_id"], "category": task["category"],
              "detected_category": detected_category, "template_used": template_used,
              "template_ok": template_ok, "gold": task["gold"], "answer": answer,
              "raw_score": raw, "corrected_score": corrected}
        results.append(row)

    router.close()

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results -> {RESULTS_OUT}\n")

    print("=" * 100)
    print("PER-TASK RESULTS (deployed path: bare {task_id, prompt} -> classify -> template -> kimi)")
    print("=" * 100)
    header = f"{'task_id':<22}{'true cat':<18}{'detected':<12}{'template ok?':<13}{'raw':>6}{'corr':>6}  answer / gold"
    print(header)
    for r in results:
        flag = "OK" if r["template_ok"] else "MISROUTE"
        print(f"{r['task_id']:<22}{r['category']:<18}{r['detected_category']:<12}{flag:<13}"
             f"{r['raw_score']:>6.2f}{r['corrected_score']:>6.2f}  "
             f"ans={r['answer'][:60]!r} gold={str(r['gold'])[:40]!r}")

    print(f"\n{len(misroutes)} template misroute(s) out of {len(tasks)}:")
    for task_id, true_cat, detected, used, expected in misroutes:
        print(f"  {task_id}: true={true_cat!r} detected={detected!r} "
             f"template_used={used!r} (expected {expected!r})")

    print("\n" + "=" * 100)
    print("PER-CATEGORY CORRECTED ACCURACY")
    print("=" * 100)
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    total_correct = total_n = 0
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        n = len(rows)
        correct = sum(1 for r in rows if r["corrected_score"] >= 0.99)
        proxy = " (token_f1 proxy -- hand-check)" if cat in PROXY_ONLY_CATEGORIES - {"sentiment"} else ""
        print(f"{cat:<18}{correct}/{n}{proxy}")
        total_correct += correct
        total_n += n
    print(f"\nOVERALL (corrected, excluding proxy-only categories from a hard pass/fail claim): "
         f"{total_correct}/{total_n} ({total_correct/total_n:.0%})")
    print("\nNOTE: summarization/entity_extraction 'correct' counts above use a >=0.99 token_f1 "
         "threshold, which is almost never met by a paraphrased answer -- read the raw answers "
         "yourself for those two categories, do not trust the count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

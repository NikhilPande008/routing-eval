#!/usr/bin/env python3
"""Accuracy diagnostic (2026-07-09, D34): runs the 32-task fixture through
BOTH minimax-m3 and kimi-k2p7-code via real Fireworks, using the SAME
per-category prompt templates the deployed policy uses (so results are
apples-to-apples with what's actually graded), scores against gold with the
declared scorer, and reports per-category accuracy for both models plus a
side-by-side raw-answer dump for hand-judging the categories where the
scorer is only a loose proxy (sentiment, summarization, entity_extraction).

This is NOT a token-cost tool -- it exists because the bake-off (D30) picked
kimi-k2p7-code on tokens alone with zero accuracy signal (the practice tasks
had no gold), and the real submission then failed the accuracy gate at
57.9% (D34). Do not add token-cost logic here; tokens come back only as a
tiebreaker AFTER a category clears the accuracy floor.

Run:
  set -a && source .env && set +a
  export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
  python3 scripts/accuracy_diagnostic.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval import scorers  # noqa: E402
from routing_eval.llm import OpenAICompatibleClient  # noqa: E402
from routing_eval.llm.runners import RemoteRunner  # noqa: E402
from routing_eval.modelids import normalize_model_id  # noqa: E402
from routing_eval.policy import DEFAULT_POLICY_PATH, _strip_code_fence  # noqa: E402
from routing_eval.prompts import load_category_templates, system_prompt  # noqa: E402
from routing_eval.schema import Item  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "accuracy_diagnostic.json")
RESULTS_OUT = "/tmp/accuracy_diagnostic_results.json"
MODELS = ["minimax-m3", "kimi-k2p7-code"]
MAX_TOKENS = 256          # matches the deployed policy's max_tokens (D27/D32) -- apples-to-apples
PROXY_ONLY_CATEGORIES = {"sentiment", "summarization", "entity_extraction"}
PASS_THRESHOLD = 0.5      # token_f1 is continuous; >=0.5 is a reasonable "roughly right" cut


def load_tasks():
    with open(FIXTURE) as f:
        return json.load(f)


def answer_one(client, model, task, category_templates):
    item = Item(
        id=task["task_id"], task_type=task["category"], input=task["prompt"],
        gold=task["gold"], scorer=task["scorer"], scorer_opts=task.get("scorer_opts") or {},
    )
    template_name = category_templates.get(task["category"], category_templates.get("_default", "default"))
    system = system_prompt(template_name)
    remote = RemoteRunner(client, model=normalize_model_id(model), max_tokens=MAX_TOKENS, system=system)
    out = remote.run(item)
    answer = out.answer
    if template_name == "code_only":
        answer = _strip_code_fence(answer)   # matches what the deployed score path does (policy.py)
    score = scorers.score(answer, item)
    return answer, score, out.total_tokens


def main() -> int:
    base_url = os.environ["FIREWORKS_BASE_URL"]
    api_key = os.environ["FIREWORKS_API_KEY"]
    client = OpenAICompatibleClient(base_url, api_key=api_key)
    category_templates = load_category_templates(DEFAULT_POLICY_PATH)

    tasks = load_tasks()
    results = []

    for task in tasks:
        row = {"task_id": task["task_id"], "category": task["category"], "gold": task["gold"],
              "note": task.get("note", ""), "models": {}}
        for model in MODELS:
            try:
                answer, score, tokens = answer_one(client, model, task, category_templates)
            except Exception as e:  # noqa: BLE001 -- diagnostic must finish even if one call fails
                print(f"diagnostic: {task['task_id']} / {model} failed -- {e}", file=sys.stderr)
                answer, score, tokens = "", 0.0, 0
            row["models"][model] = {"answer": answer, "score": score, "tokens": tokens}
        results.append(row)
        print(f"\n=== {row['task_id']} ({row['category']}) ===")
        print(f"gold: {row['gold']!r}")
        for model in MODELS:
            m = row["models"][model]
            print(f"  {model:<16} score={m['score']:.2f}  answer: {m['answer'][:200]!r}")

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results -> {RESULTS_OUT}")

    print("\n" + "=" * 78)
    print(f"PER-CATEGORY ACCURACY (mean score / pass-rate @ {PASS_THRESHOLD:.0%})")
    print("=" * 78)
    by_cat = defaultdict(list)
    for row in results:
        by_cat[row["category"]].append(row)

    kimi_wrong = []
    header = (f"{'category':<18}{'kimi mean':>10}{'kimi pass':>11}"
             f"{'minimax mean':>14}{'minimax pass':>14}  {'winner':<8}proxy-only?")
    print(header)
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        kimi_scores = [r["models"]["kimi-k2p7-code"]["score"] for r in rows]
        mm_scores = [r["models"]["minimax-m3"]["score"] for r in rows]
        kimi_mean = sum(kimi_scores) / len(kimi_scores)
        mm_mean = sum(mm_scores) / len(mm_scores)
        kimi_pass = sum(1 for s in kimi_scores if s >= PASS_THRESHOLD)
        mm_pass = sum(1 for s in mm_scores if s >= PASS_THRESHOLD)
        winner = "kimi" if kimi_mean > mm_mean else ("minimax" if mm_mean > kimi_mean else "tie")
        proxy = "YES -- hand-judge" if cat in PROXY_ONLY_CATEGORIES else ""
        print(f"{cat:<18}{kimi_mean:>10.2f}{kimi_pass:>7}/{len(rows):<3}"
             f"{mm_mean:>14.2f}{mm_pass:>11}/{len(rows):<3}  {winner:<8}{proxy}")

        for r in rows:
            if r["models"]["kimi-k2p7-code"]["score"] < PASS_THRESHOLD:
                kimi_wrong.append(r)

    print(f"\n{len(kimi_wrong)} task(s) where kimi-k2p7-code (the currently DEPLOYED model) "
         f"scored below {PASS_THRESHOLD:.0%}:")
    for r in kimi_wrong:
        print(f"  {r['task_id']} ({r['category']}): score={r['models']['kimi-k2p7-code']['score']:.2f}")
        print(f"    answer: {r['models']['kimi-k2p7-code']['answer'][:150]!r}")
        print(f"    gold:   {r['gold']!r}")

    print("\nNOTE: sentiment/summarization/entity_extraction scores are a token_f1 PROXY only "
         "-- do not trust the pass/fail call for these categories without reading the raw "
         "answers above yourself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

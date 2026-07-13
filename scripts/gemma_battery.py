#!/usr/bin/env python3
"""Gemma-line battery (2026-07-12, second-account submission). ONE invocation
that runs the 10 official validation tasks (scripts/fixtures/official_tasks.json,
each with its Expected rubric) + the 8 real practice tasks through the GEMMA
policy (routing_policy.gemma.json -- remote = Gemma with per-call kimi
fallback, local tier as-is) against a live endpoint, and reports per-task
answers, rubric-proxy verdicts, and token totals, cached to a records JSON.

Judge-proxy is DEV-token spend, never scored. Official tasks are graded
against their rubric text VERBATIM (judge_proxy_diagnostic.judge(rubric=...));
practice tasks have no rubric, so they get the generic 'fulfills the task?'
judge.

Env:
  FIREWORKS_BASE_URL, FIREWORKS_API_KEY   -- account-2 Fireworks creds
  ALLOWED_MODELS                          -- account-2 allowed models (kimi fallback lives here)
  GEMMA_MODEL_ID                          -- the Gemma model id (unset => whole line degrades to kimi, logged)
  LOCAL_BASE_URL / LOCAL_MODEL (optional) -- enable the local tier; omit for remote-only
  POLICY_PATH (optional)                  -- override the policy file (default: the gemma policy)

Run:
  set -a && source .env.account2 && set +a
  export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
  export GEMMA_MODEL_ID=<gemma-model-id>
  python3 scripts/gemma_battery.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval.llm import OpenAICompatibleClient  # noqa: E402
from routing_eval.llm.runners import LocalRunner  # noqa: E402
from routing_eval.modelids import split_models  # noqa: E402
from routing_eval.policy import PolicyRouter, load_policy  # noqa: E402
from routing_eval.taskio import load_tasks, task_id as _task_id, task_prompt as _task_prompt  # noqa: E402
from scripts.judge_proxy_diagnostic import answer_with_diagnostics, judge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GEMMA_POLICY = os.path.join(os.path.dirname(HERE), "routing_eval", "routing_policy.gemma.json")
OFFICIAL_FIXTURE = os.path.join(HERE, "fixtures", "official_tasks.json")
PRACTICE_FIXTURE = os.path.join(HERE, "fixtures", "practice_tasks.json")
RECORDS_OUT = os.environ.get("GEMMA_BATTERY_OUT", "/tmp/gemma_battery_records.json")


def load_battery():
    """10 official tasks (with rubric) + 8 practice tasks (no rubric).
    Returns dicts: {task_id, prompt, rubric?, source}."""
    tasks = []
    with open(OFFICIAL_FIXTURE) as f:
        for t in json.load(f):
            tasks.append({"task_id": t["task_id"], "prompt": t["prompt"],
                         "rubric": t.get("rubric"), "gold": t.get("gold"),
                         "category": t.get("category"), "source": "official"})
    for i, t in enumerate(load_tasks(PRACTICE_FIXTURE)):
        tasks.append({"task_id": _task_id(t, i), "prompt": _task_prompt(t, i),
                     "rubric": None, "source": "practice"})
    return tasks


def build_router():
    base_url = os.environ["FIREWORKS_BASE_URL"]
    api_key = os.environ["FIREWORKS_API_KEY"]
    allowed_models = split_models(os.environ["ALLOWED_MODELS"])
    client = OpenAICompatibleClient(base_url, api_key=api_key)
    policy_path = os.environ.get("POLICY_PATH") or GEMMA_POLICY
    print(f"gemma_battery: policy={policy_path}", file=sys.stderr)
    if not (os.environ.get("GEMMA_MODEL_ID") or "").strip():
        print("gemma_battery: WARNING GEMMA_MODEL_ID is unset -- every Gemma call "
              "will resolve straight to the kimi fallback (accuracy-safe, but this "
              "is NOT a real Gemma run)", file=sys.stderr)
    # FORCE_REMOTE_ONLY: an explicit override that WINS over LOCAL_BASE_URL,
    # so a stray local server left configured in the shell can't silently
    # reintroduce the local tier. Use this to measure Gemma's own per-category
    # competence in isolation -- every category (all 8, including the ones
    # routing_policy.gemma.json marks tier="local") goes straight to remote-
    # Gemma, since local=None short-circuits PolicyRouter's local branch
    # entirely (route_task's `if entry.tier == "local" and self.local is not
    # None` is False either way -- this is belt-and-suspenders, not a new
    # code path).
    force_remote = (os.environ.get("FORCE_REMOTE_ONLY", "").strip().lower()
                    not in ("", "0", "false"))
    local = None
    if force_remote:
        print("gemma_battery: FORCE_REMOTE_ONLY=1 -- local tier DISABLED for this run "
             "(overrides LOCAL_BASE_URL if set). All 8 categories forced remote-Gemma: "
             "measuring Gemma's own competence, not the composed local+remote policy.",
             file=sys.stderr)
    elif os.environ.get("LOCAL_BASE_URL"):
        local = LocalRunner(OpenAICompatibleClient(os.environ["LOCAL_BASE_URL"]),
                            model=os.environ.get("LOCAL_MODEL", "local-model"),
                            max_tokens=int(os.environ.get("LOCAL_MAX_TOKENS", "256")),
                            logprobs=False)
        print(f"gemma_battery: local tier ENABLED via {os.environ['LOCAL_BASE_URL']}",
              file=sys.stderr)
    else:
        print("gemma_battery: local tier DISABLED (no LOCAL_BASE_URL) -- local-tier "
              "categories escalate to Gemma-remote", file=sys.stderr)
    return PolicyRouter(policy=load_policy(policy_path), remote_client=client,
                        allowed_models=allowed_models, local=local), client


def main() -> int:
    router, client = build_router()
    tasks = load_battery()
    results = []
    answer_tokens = judge_tokens = 0

    for t in tasks:
        category, template_used, attempt = answer_with_diagnostics(
            router, t["task_id"], t["prompt"])
        verdict, judge_text, jt = judge(client, t["prompt"], attempt.answer,
                                        rubric=t.get("rubric"))
        answer_tokens += attempt.total_tokens
        judge_tokens += jt
        row = {"task_id": t["task_id"], "source": t["source"],
              "category": t.get("category"), "detected_category": category,
              "template_used": template_used, "rubric": t.get("rubric"),
              "gold": t.get("gold"), "answer": attempt.answer,
              "finish_reason": attempt.finish_reason,
              "answer_tokens": attempt.total_tokens,
              "judge_verdict": verdict, "judge_reason": judge_text, "judge_tokens": jt}
        results.append(row)
        graded = "rubric" if t.get("rubric") else "generic"
        print(f"{t['task_id']:<10} {t['source']:<9} cat={category:<18} "
             f"tmpl={template_used:<30} verdict={verdict:<6} ({graded}) "
             f"ans_tok={attempt.total_tokens}")
        print(f"    answer: {attempt.answer[:160]!r}")
        print(f"    judge:  {judge_text[:160]!r}")

    router.close()
    with open(RECORDS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRecords -> {RECORDS_OUT}\n")

    print("=" * 90)
    print("RUBRIC-PROXY PASS RATE -- OFFICIAL TASKS (graded verbatim against each rubric)")
    print("=" * 90)
    official = [r for r in results if r["source"] == "official"]
    by_cat = defaultdict(list)
    for r in official:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        p = sum(1 for r in rows if r["judge_verdict"] == "PASS")
        print(f"{cat:<26}{p}/{len(rows)}  ({', '.join(r['task_id'] for r in rows)})")
    op = sum(1 for r in official if r["judge_verdict"] == "PASS")
    print(f"\nOFFICIAL TOTAL: {op}/{len(official)}")

    practice = [r for r in results if r["source"] == "practice"]
    pp = sum(1 for r in practice if r["judge_verdict"] == "PASS")
    print(f"PRACTICE TOTAL (generic judge): {pp}/{len(practice)}")

    print("\n" + "=" * 90)
    print("TOKENS")
    print("=" * 90)
    n = len(results)
    print(f"Answer tokens (scored-equivalent): {answer_tokens} across {n} tasks "
         f"-- {answer_tokens / n:.1f}/task")
    print(f"Judge tokens (DEV-ONLY, never scored): {judge_tokens} across {n} judge calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

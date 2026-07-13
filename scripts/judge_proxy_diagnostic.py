#!/usr/bin/env python3
"""D40 diagnostic: a local judge-proxy that approximates the real external
LLM judge (D18) far better than our own scorers.py ever could -- for each
answered task, a SECOND call (to minimax-m3, deliberately a different model
than the answerer) asks "does this answer correctly and completely fulfill
what the task asked?" and gets a PASS/FAIL + one-line reason back. This is
dev-token spend (never touches the submission), not scored spend.

Runs the FULL deployed path (classify -> policy lookup -> template -> real
Fireworks call, including D40's finish_reason-triggered length-retry) over
BOTH the 32-task accuracy diagnostic AND the 8 real practice tasks -- 40
tasks total. Each task is handed in as bare {task_id, prompt} (no category
field), exactly the real /input/tasks.json shape (D24), so classification is
genuinely blind, same as at grading time.

This intentionally calls PolicyRouter's internal building blocks
(_resolve_model, _try_fireworks, _pick_retry_model) directly rather than
route_task(), so it can capture per-task token/finish_reason data that
route_task() deliberately does NOT return (the real {task_id, answer}
output contract, D20, must not grow extra fields). The logic replicated here
is the SAME sequence route_task() runs internally (classify -> resolve_entry
-> primary attempt -> blank-answer retry with a different model) -- no
prompt or classification logic is duplicated, only orchestrated inline for
visibility.

Run:
  set -a && source .env && set +a
  export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
  python3 scripts/judge_proxy_diagnostic.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval.llm import OpenAICompatibleClient  # noqa: E402
from routing_eval.llm.runners import LocalRunner, RemoteRunner  # noqa: E402
from routing_eval.localcheck import local_answer_problem  # noqa: E402
from routing_eval.modelids import normalize_model_id, split_models  # noqa: E402
from routing_eval.policy import (DEFAULT_POLICY_PATH, PolicyRouter, _resolve_model,  # noqa: E402
                                 _system_prompt, load_policy, resolve_entry)
from routing_eval.schema import Item  # noqa: E402
from routing_eval.taskio import load_tasks, task_id as _task_id, task_prompt as _task_prompt  # noqa: E402

DIAGNOSTIC_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "accuracy_diagnostic.json")
PRACTICE_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "practice_tasks.json")
RESULTS_OUT = "/tmp/judge_proxy_results.json"
JUDGE_MODEL = "minimax-m3"          # deliberately a different model than the answerer (kimi)
JUDGE_MAX_TOKENS = 128

JUDGE_SYSTEM = (
    "You are grading whether an AI assistant's answer correctly and "
    "completely fulfills what a task asked. Read the TASK and the ANSWER "
    "below. Reply with exactly two lines: the first line is exactly PASS or "
    "FAIL, the second line is a one-sentence reason."
)


def load_all_tasks():
    """32 diagnostic tasks (true category known) + 8 real practice tasks
    (no category field, D24 -- exactly the real contract). Returns a list of
    dicts: {task_id, prompt, true_category (or None), source}."""
    tasks = []
    with open(DIAGNOSTIC_FIXTURE) as f:
        for t in json.load(f):
            tasks.append({"task_id": t["task_id"], "prompt": t["prompt"],
                         "true_category": t["category"], "source": "diagnostic"})
    for i, t in enumerate(load_tasks(PRACTICE_FIXTURE)):
        tasks.append({"task_id": _task_id(t, i), "prompt": _task_prompt(t, i),
                     "true_category": None, "source": "practice"})
    return tasks


def answer_with_diagnostics(router: PolicyRouter, task_id: str, prompt: str):
    """Replicates route_task()'s real sequence (classify -> resolve_entry ->
    primary attempt -> blank-answer retry with a different model), using
    PolicyRouter's own methods, but returns the FireworksAttempt (answer +
    finish_reason + tokens) route_task() itself doesn't expose externally."""
    cls = router.classifier.classify(prompt)
    entry = resolve_entry(router.policy, cls.category)
    item = Item(id=task_id, task_type=cls.category, input=prompt, gold=None, scorer="exact")

    # 2026-07-11 (local tier): same sequence route_task() runs -- local call
    # first when the entry is local-tier and a local runner exists, kept only
    # if localcheck passes it; ANY problem escalates to the Fireworks path
    # below. A kept local answer reports 0 answer tokens (D26: local answers
    # are scored at zero token cost).
    if entry.tier == "local" and router.local is not None:
        timeout_s = entry.timeout_s or router.default_timeout_s
        local_system = _system_prompt(entry.local_prompt_template or entry.prompt_template)
        try:
            out = router.local.run(item, timeout_s, local_system)
            problem = local_answer_problem(cls.category, prompt, out.answer)
        except Exception as e:  # noqa: BLE001 -- mirror route_task: degrade to Fireworks
            out, problem = None, f"local call failed: {e}"
        if problem is None:
            from routing_eval.policy import FireworksAttempt
            tmpl = entry.local_prompt_template or entry.prompt_template
            return cls.category, tmpl + " (LOCAL)", FireworksAttempt(out.answer, "local", 0)
        print(f"    local answer rejected ({problem}) -- escalating", file=sys.stderr)

    primary_model = _resolve_model(entry, router.allowed_models)
    attempt = router._try_fireworks(entry, item, primary_model)
    if not attempt.answer or not attempt.answer.strip():
        retry_model = router._pick_retry_model(primary_model)
        if retry_model is not None:
            attempt = router._try_fireworks(entry, item, retry_model)
    return cls.category, entry.prompt_template, attempt


def judge(client, task_prompt: str, answer: str):
    judge_input = (
        f"TASK:\n{task_prompt}\n\nANSWER:\n{answer}\n\n"
        "Does this answer correctly and completely fulfill what the task asked? "
        "Reply PASS or FAIL with one line why."
    )
    item = Item(id="judge", task_type="judge", input=judge_input, gold=None, scorer="exact")
    remote = RemoteRunner(client, model=normalize_model_id(JUDGE_MODEL),
                          max_tokens=JUDGE_MAX_TOKENS, system=JUDGE_SYSTEM)
    out = remote.run(item, timeout=30.0)
    text = out.answer.strip()
    upper = text.upper()
    if upper.startswith("PASS"):
        verdict = "PASS"
    elif upper.startswith("FAIL"):
        verdict = "FAIL"
    else:
        verdict = "UNKNOWN"
    return verdict, text, out.total_tokens


def main() -> int:
    base_url = os.environ["FIREWORKS_BASE_URL"]
    api_key = os.environ["FIREWORKS_API_KEY"]
    allowed_models = split_models(os.environ["ALLOWED_MODELS"])
    client = OpenAICompatibleClient(base_url, api_key=api_key)
    policy = load_policy(DEFAULT_POLICY_PATH)
    local = None
    local_base_url = os.environ.get("LOCAL_BASE_URL")
    if local_base_url:
        local = LocalRunner(OpenAICompatibleClient(local_base_url),
                            model=os.environ.get("LOCAL_MODEL", "local-model"),
                            logprobs=False)
        print(f"local tier ENABLED via {local_base_url}", file=sys.stderr)
    else:
        print("local tier DISABLED (no LOCAL_BASE_URL) -- local-tier categories "
              "will all escalate to Fireworks", file=sys.stderr)
    router = PolicyRouter(policy=policy, remote_client=client, allowed_models=allowed_models,
                          local=local)

    tasks = load_all_tasks()
    results = []
    answer_tokens_total = 0
    judge_tokens_total = 0

    for t in tasks:
        category, template_used, attempt = answer_with_diagnostics(router, t["task_id"], t["prompt"])
        verdict, judge_text, judge_tokens = judge(client, t["prompt"], attempt.answer)
        answer_tokens_total += attempt.total_tokens
        judge_tokens_total += judge_tokens

        row = {"task_id": t["task_id"], "source": t["source"], "true_category": t["true_category"],
              "detected_category": category, "template_used": template_used,
              "answer": attempt.answer, "finish_reason": attempt.finish_reason,
              "answer_tokens": attempt.total_tokens, "judge_verdict": verdict,
              "judge_reason": judge_text, "judge_tokens": judge_tokens}
        results.append(row)
        print(f"{t['task_id']:<24} cat={category:<10} tmpl={template_used:<28} "
             f"verdict={verdict:<8} ans_tok={attempt.total_tokens:<5} "
             f"finish={attempt.finish_reason!r}")
        print(f"    answer: {attempt.answer[:150]!r}")
        print(f"    judge:  {judge_text[:150]!r}")

    router.close()

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results -> {RESULTS_OUT}\n")

    print("=" * 100)
    print("JUDGE-PROXY PASS RATE, BY TRUE CATEGORY (diagnostic tasks only, 32)")
    print("=" * 100)
    by_cat = defaultdict(list)
    for r in results:
        if r["source"] == "diagnostic":
            by_cat[r["true_category"]].append(r)
    diag_pass = diag_total = 0
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        n = len(rows)
        passed = sum(1 for r in rows if r["judge_verdict"] == "PASS")
        diag_pass += passed
        diag_total += n
        print(f"{cat:<18}{passed}/{n}")
    print(f"\nDIAGNOSTIC TOTAL: {diag_pass}/{diag_total} ({diag_pass/diag_total:.1%})")

    print("\n" + "=" * 100)
    print("JUDGE-PROXY PASS RATE, PRACTICE TASKS (8, no ground-truth category -- grouped by "
         "DETECTED category)")
    print("=" * 100)
    practice_rows = [r for r in results if r["source"] == "practice"]
    by_detected = defaultdict(list)
    for r in practice_rows:
        by_detected[r["detected_category"]].append(r)
    prac_pass = len(practice_rows) and sum(1 for r in practice_rows if r["judge_verdict"] == "PASS")
    for cat in sorted(by_detected):
        rows = by_detected[cat]
        n = len(rows)
        passed = sum(1 for r in rows if r["judge_verdict"] == "PASS")
        print(f"{cat:<18}{passed}/{n}")
    print(f"\nPRACTICE TOTAL: {prac_pass}/{len(practice_rows)}")

    truncated = [r for r in results if r["finish_reason"] == "length"]
    print(f"\n{len(truncated)}/{len(results)} tasks still truncated (finish_reason='length') "
         f"even after the doubled-cap retry:")
    for r in truncated:
        print(f"  {r['task_id']}: {r['answer'][:80]!r}")

    print("\n" + "=" * 100)
    print("TOKEN TOTALS")
    print("=" * 100)
    print(f"Answer tokens (what would count toward the real token score): {answer_tokens_total} "
         f"across {len(results)} tasks -- {answer_tokens_total/len(results):.1f}/task avg")
    print(f"Judge tokens (dev-only, local diagnostic -- NEVER part of a real submission): "
         f"{judge_tokens_total} across {len(results)} judge calls -- "
         f"{judge_tokens_total/len(results):.1f}/task avg")
    print(f"Combined dev-session spend (answer + judge): {answer_tokens_total + judge_tokens_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

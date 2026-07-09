"""CLI, six commands:
  run             dataset -> records.json     (mock models now; real models are P2)
  frontier        records.json -> report      (free; sweeps threshold, compares gates)
  score           /input/tasks.json -> /output/results.json  (the scoring entrypoint;
                  classify -> policy -> local/Fireworks call, see routing_eval.policy)
  probe-local     local-tier viability probe, per category (see routing_eval.modelselect)
  bakeoff         Fireworks model bake-off over ALLOWED_MODELS, per category
  generate-policy draft routing_policy.json from probe-local + bakeoff output
"""
from __future__ import annotations

import argparse
import os

from .classify import KeywordClassifier
from .conformance import DEFAULT_INPUT, DEFAULT_OUTPUT, load_tasks, split_models
from .conformance import run as run_conformance
from .datasets import make_classification, make_math, make_qa, make_standin
from .frontier import evaluate
from .llm import OpenAICompatibleClient, StubClient, stub_response
from .llm.runners import LocalRunner
from .mock import DEFAULT_LOCAL, build_records
from .modelselect import (DEFAULT_LATENCY_BUDGET_S, assess_local_viability,
                          load_bakeoff_records, load_task_results, rank_models_by_category,
                          run_bakeoff, run_local_probe, save_bakeoff_records,
                          save_task_results)
from .policy import DEFAULT_POLICY_PATH, generate_policy, save_policy
from .prompts import load_category_templates
from .report import (ascii_plot, format_bakeoff_ranking, format_local_viability,
                     summarize, to_csv)
from .schema import load_records, save_records

_DATASETS = {"standin": make_standin, "math": make_math,
             "classification": make_classification, "qa": make_qa}
DEFAULT_TASKS = "scripts/fixtures/practice_tasks.json"


def _run(a):
    items = _DATASETS[a.dataset](a.n, a.seed)
    comp = dict(DEFAULT_LOCAL)
    if a.local_competence is not None:
        for tier, v in zip(("easy", "borderline", "hard"), a.local_competence):
            comp[tier] = v
    recs = build_records(items, local_competence=comp, calib_noise=a.calib_noise,
                         remote_competence=a.remote_competence, seed=a.seed)
    save_records(recs, a.out)
    print(f"wrote {len(recs)} records -> {a.out}  "
          f"(mock models; real vLLM/Fireworks runners are P2)")


def _frontier(a):
    recs = load_records(a.records)
    signals = a.signals or sorted(recs[0].confidences.keys())
    best = None
    for s in signals:
        fr = evaluate(recs, s, a.accuracy_threshold)
        print(summarize(fr))
        print()
        if fr.feasible and (best is None or
                            fr.operating_point.remote_tokens < best.operating_point.remote_tokens):
            best = fr
    if best is not None:
        print("=" * 48)
        print(f"BEST gate: '{best.signal}' -> {best.operating_point.remote_tokens} remote "
              f"tokens at acc {best.operating_point.accuracy:.3f} "
              f"(floor {a.accuracy_threshold:.3f})")
        print(ascii_plot(best))
        if a.csv_out:
            to_csv(best, a.csv_out)
            print(f"\nfrontier CSV -> {a.csv_out}")
    else:
        print("No gate is feasible at this floor on this set.")


def _score(a):
    raise SystemExit(run_conformance(a.input, a.output, policy_path=a.policy))


def _probe_local(a):
    tasks = load_tasks(a.tasks)
    if a.local_base_url:
        client = OpenAICompatibleClient(a.local_base_url)
    else:
        client = StubClient(lambda kw: stub_response(
            [f"stub-local:{kw['messages'][-1]['content'][:40]}"],
            usage={"prompt_tokens": 0, "completion_tokens": 4, "total_tokens": 4}))
        print("probe-local: no --local-base-url given, using a trivial stub client "
              "(point it at a real local GGUF server when you have one)")
    local = LocalRunner(client, model=a.local_model, max_tokens=a.max_tokens)
    results = run_local_probe(local, tasks)
    viability = assess_local_viability(results, a.accuracy_floor, a.latency_budget_s)
    print(format_local_viability(viability))
    if a.out:
        save_task_results(results, a.out)
        print(f"\ntask results -> {a.out}")


def _bakeoff(a):
    tasks = load_tasks(a.tasks)
    if a.use_cache:
        per_model = load_bakeoff_records(a.out)
    else:
        base_url = os.environ["FIREWORKS_BASE_URL"]
        api_key = os.environ.get("FIREWORKS_API_KEY", "")
        models = split_models(os.environ["ALLOWED_MODELS"])
        client = OpenAICompatibleClient(base_url, api_key=api_key)
        classifier = None if a.no_classify else KeywordClassifier()
        category_templates = (None if a.no_classify
                              else load_category_templates(a.policy or DEFAULT_POLICY_PATH))
        per_model, _ = run_bakeoff(tasks, models, client, a.accuracy_floor, a.max_tokens,
                                   classifier=classifier, category_templates=category_templates)
        save_bakeoff_records(per_model, a.out)
        print(f"bake-off records -> {a.out}  (record-then-replay: re-run with --use-cache "
              f"to re-rank for free)")
    ranking = rank_models_by_category(tasks, per_model, a.accuracy_floor)
    print(format_bakeoff_ranking(ranking))


def _generate_policy(a):
    tasks = load_tasks(a.tasks)

    viability = {}
    if a.probe_results:
        results = load_task_results(a.probe_results)
        viability = assess_local_viability(results, a.accuracy_floor, a.latency_budget_s)

    ranking = {}
    if a.bakeoff_records:
        per_model = load_bakeoff_records(a.bakeoff_records)
        ranking = rank_models_by_category(tasks, per_model, a.accuracy_floor)

    if not viability and not ranking:
        print("generate-policy: neither --probe-results nor --bakeoff-records given -- "
              "nothing to calibrate from. Run probe-local and/or bakeoff first.")
        raise SystemExit(1)

    policy = generate_policy(viability, ranking, a.max_tokens, a.prompt_template)
    save_policy(policy, a.out)
    print(f"draft routing policy -> {a.out}  "
          f"(review before replacing the checked-in default at {DEFAULT_POLICY_PATH})")
    for cat, entry in policy.items():
        print(f"  {cat:<18} tier={entry['tier']:<10} model={entry['model']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="routing-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate records (mock models)")
    r.add_argument("--dataset", choices=list(_DATASETS), default="standin")
    r.add_argument("--n", type=int, default=100, help="items per task (standin) or total")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="records.json")
    r.add_argument("--local-competence", type=float, nargs=3, default=None,
                   metavar=("EASY", "BORDER", "HARD"),
                   help="mock local accuracy per difficulty tier")
    r.add_argument("--calib-noise", type=float, default=0.6,
                   help="gate calibration noise; lower = better-calibrated gate")
    r.add_argument("--remote-competence", type=float, default=0.92)
    r.set_defaults(fn=_run)

    f = sub.add_parser("frontier", help="trace frontier + locate operating point")
    f.add_argument("--records", default="records.json")
    f.add_argument("--accuracy-threshold", type=float, required=True)
    f.add_argument("--signals", nargs="*", default=None,
                   help="which confidence signals to compare (default: all)")
    f.add_argument("--csv-out", default=None)
    f.set_defaults(fn=_frontier)

    s = sub.add_parser("score", help="/input/tasks.json -> /output/results.json "
                                     "(classify -> policy -> local/Fireworks)")
    s.add_argument("--input", default=DEFAULT_INPUT)
    s.add_argument("--output", default=DEFAULT_OUTPUT)
    s.add_argument("--policy", default=None,
                   help=f"routing_policy.json path (default: checked-in safe default, "
                        f"{DEFAULT_POLICY_PATH})")
    s.set_defaults(fn=_score)

    pl = sub.add_parser("probe-local", help="local-tier viability probe, per category")
    pl.add_argument("--tasks", default=DEFAULT_TASKS)
    pl.add_argument("--local-base-url", default=None,
                    help="OpenAI-compatible local server (e.g. a GGUF server); "
                         "omit to use a trivial stub client")
    pl.add_argument("--local-model", default="local-stub")
    pl.add_argument("--max-tokens", type=int, default=64)
    pl.add_argument("--accuracy-floor", type=float, default=0.80)
    pl.add_argument("--latency-budget-s", type=float, default=DEFAULT_LATENCY_BUDGET_S)
    pl.add_argument("--out", default=None, help="optional: save per-task results here")
    pl.set_defaults(fn=_probe_local)

    b = sub.add_parser("bakeoff", help="Fireworks model bake-off over ALLOWED_MODELS, per category")
    b.add_argument("--tasks", default=DEFAULT_TASKS)
    b.add_argument("--accuracy-floor", type=float, default=0.80)
    b.add_argument("--max-tokens", type=int, default=64)
    b.add_argument("--out", default="bakeoff_records.json",
                   help="record-then-replay cache: model -> per-task results")
    b.add_argument("--use-cache", action="store_true",
                   help="skip live calls; re-rank from --out (free, per D3)")
    b.add_argument("--policy", default=None,
                   help="routing_policy.json to read per-category prompt templates from "
                        f"(default: checked-in safe default, {DEFAULT_POLICY_PATH})")
    b.add_argument("--no-classify", action="store_true",
                   help="skip classification; use each task dict's own 'category' field "
                        "(or 'uncategorized') and the generic prompt for every call")
    b.set_defaults(fn=_bakeoff)

    g = sub.add_parser("generate-policy",
                       help="draft routing_policy.json from probe-local + bakeoff output")
    g.add_argument("--tasks", default=DEFAULT_TASKS)
    g.add_argument("--probe-results", default=None,
                   help="probe-local --out path (per-task local results)")
    g.add_argument("--bakeoff-records", default=None,
                   help="bakeoff --out path (model -> per-task results)")
    g.add_argument("--accuracy-floor", type=float, default=0.80)
    g.add_argument("--latency-budget-s", type=float, default=DEFAULT_LATENCY_BUDGET_S)
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--prompt-template", default="default")
    g.add_argument("--out", default="routing_policy.draft.json")
    g.set_defaults(fn=_generate_policy)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

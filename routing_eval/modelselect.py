"""Model-selection measurement rig for Track 1.

Two parts sharing one per-category rollup (`evaluate_by_category`):

  A. Local-tier viability probe (`run_local_probe` + `assess_local_viability`):
     for a candidate local model, measures per-category accuracy and CPU
     wall-clock latency, and buckets each category as local-viable (fast +
     likely-correct -> costs zero tokens per DECISIONS.md D17) or
     must-escalate (bounded by D19's 30s/request budget).
  B. Fireworks model bake-off (`run_bakeoff`): given ALLOWED_MODELS at
     runtime, runs each allowed model over the task set once (record-then-
     replay -- same run-once-cache-forever pattern as schema.save_records/
     load_records, D3), scores with the same metric, and ranks models per
     category by total prompt+completion tokens among those that clear the
     accuracy floor.

No model ID is hardcoded anywhere in this file. Part A takes an injected
LocalRunner (point it at a real local GGUF server via OpenAICompatibleClient
when one exists; a StubClient is a legitimate trivial default today). Part B
reads ALLOWED_MODELS from the environment. Both are proven now against a
stub Fireworks server (scripts/fake_fireworks_server.py); swapping in a real
local model or real ALLOWED_MODELS on launch day is a config change, not a
code change.

Correctness proxy: the competition's real accuracy gate is an external
LLM-judge (D18) whose rubric we don't control. Until that judge (or a proxy
for it) is exposed, this rig scores answers the same way the rest of the repo
already does -- routing_eval.scorers against each task's own `gold`+`scorer`
fields -- via a `schema.Item` built from the task dict. That is a stand-in,
not the real judge; treat accuracy numbers from this rig as a plumbing proof,
not a competition-accuracy claim (see DECISIONS.md D8).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import scorers
from .classify import Classifier
from .llm.runners import LocalRunner, RemoteRunner
from .prompts import system_prompt as _system_prompt
from .schema import Item
from .taskio import task_id as _task_id
from .taskio import task_prompt as _task_prompt

DEFAULT_LATENCY_BUDGET_S = 30.0  # D19: per-request budget on the grading VM


@dataclass
class TaskResult:
    task_id: str
    category: str
    answer: str
    score: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: Optional[float] = None   # local-probe only; not a billed/budgeted remote figure


@dataclass
class CategoryStats:
    category: str
    n: int
    accuracy: float
    avg_latency_s: Optional[float]
    total_tokens: int
    avg_tokens: float


@dataclass
class LocalViability:
    category: str
    n: int
    accuracy: float
    avg_latency_s: Optional[float]
    max_latency_s: Optional[float]
    local_viable: bool
    reason: str


@dataclass
class ModelCategoryRanking:
    category: str
    model: str
    accuracy: float
    total_tokens: int
    avg_tokens: float
    clears_floor: bool


def _category(task: Dict[str, Any]) -> str:
    return str(task.get("category", "uncategorized"))


def _item_from_task(task: Dict[str, Any], idx: int) -> Item:
    """Reuses conformance's task_id/prompt fallback chain so both the scoring
    shell and this rig agree on what a "task" looks like."""
    task_id = _task_id(task, idx)
    prompt = _task_prompt(task, idx)
    return Item(
        id=task_id,
        task_type=_category(task),
        input=prompt,
        gold=task.get("gold"),
        scorer=task.get("scorer", "exact"),
        scorer_opts=task.get("scorer_opts") or {},
        difficulty=task.get("difficulty", "unknown"),
        allowed=task.get("allowed"),
    )


def evaluate_by_category(results: List[TaskResult]) -> Dict[str, CategoryStats]:
    """The shared per-category rollup Part B extends across multiple models."""
    by_cat: Dict[str, List[TaskResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    out: Dict[str, CategoryStats] = {}
    for cat, rs in by_cat.items():
        n = len(rs)
        lat = [r.latency_s for r in rs if r.latency_s is not None]
        out[cat] = CategoryStats(
            category=cat, n=n,
            accuracy=sum(r.score for r in rs) / n,
            avg_latency_s=(sum(lat) / len(lat)) if lat else None,
            total_tokens=sum(r.total_tokens for r in rs),
            avg_tokens=sum(r.total_tokens for r in rs) / n,
        )
    return out


# ---------------------------------------------------------------------------
# Part A: local-tier viability probe
# ---------------------------------------------------------------------------

def run_local_probe(local: LocalRunner, tasks: List[Dict[str, Any]]) -> List[TaskResult]:
    """Runs the candidate local model once per task, timing wall-clock
    latency around each call. Local tokens are free toward the score (D17)
    -- `total_tokens` here is a latency/diagnostic proxy, not a cost."""
    results = []
    for i, t in enumerate(tasks):
        item = _item_from_task(t, i)
        start = time.perf_counter()
        out = local.run(item)
        elapsed = time.perf_counter() - start
        score = scorers.score(out.answer, item)
        results.append(TaskResult(
            task_id=item.id, category=item.task_type, answer=out.answer, score=score,
            prompt_tokens=0, completion_tokens=out.tokens, total_tokens=out.tokens,
            latency_s=elapsed))
    return results


def assess_local_viability(results: List[TaskResult], accuracy_floor: float,
                           latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S,
                           ) -> Dict[str, LocalViability]:
    """local-viable = fast (max latency in-category <= budget) AND
    likely-correct (category accuracy >= floor). Anything else must-escalate
    -- that decision is what determines which categories cost zero tokens."""
    stats = evaluate_by_category(results)
    by_cat: Dict[str, List[TaskResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    out: Dict[str, LocalViability] = {}
    for cat, s in stats.items():
        lats = [r.latency_s for r in by_cat[cat] if r.latency_s is not None]
        max_lat = max(lats) if lats else None
        acc_ok = s.accuracy >= accuracy_floor
        lat_ok = max_lat is not None and max_lat <= latency_budget_s
        if acc_ok and lat_ok:
            reason = "fast + likely-correct"
        elif not acc_ok and not lat_ok:
            reason = f"fails accuracy floor ({s.accuracy:.2f} < {accuracy_floor:.2f}) and " \
                     f"latency budget ({max_lat:.1f}s > {latency_budget_s:.0f}s)"
        elif not acc_ok:
            reason = f"fails accuracy floor ({s.accuracy:.2f} < {accuracy_floor:.2f})"
        else:
            reason = f"exceeds {latency_budget_s:.0f}s/request budget ({max_lat:.1f}s)"
        out[cat] = LocalViability(
            category=cat, n=s.n, accuracy=s.accuracy, avg_latency_s=s.avg_latency_s,
            max_latency_s=max_lat, local_viable=acc_ok and lat_ok, reason=reason)
    return out


# ---------------------------------------------------------------------------
# Part B: Fireworks model bake-off
# ---------------------------------------------------------------------------

def run_model_over_tasks(client, model: str, tasks: List[Dict[str, Any]],
                         max_tokens: int = 64,
                         classifier: Optional[Classifier] = None,
                         category_templates: Optional[Dict[str, str]] = None,
                         ) -> List[TaskResult]:
    """One Fireworks call per task. If `classifier` is given, it (not the
    task dict's own possibly-absent "category" field) decides each item's
    category -- the real practice tasks carry no category at all, so without
    this every task silently falls into "uncategorized" and the bake-off
    can't tell categories apart. If `category_templates` is also given
    (category -> prompt_template name, e.g. from `prompts.
    load_category_templates(DEFAULT_POLICY_PATH)`), each call uses the SAME
    system prompt the real `score` path would use for that category --
    otherwise every call uses the generic default, which understates cost
    for categories with a longer template (code_only, sentiment_with_
    justification) and makes the bake-off not apples-to-apples with a real
    submission."""
    results = []
    for i, t in enumerate(tasks):
        item = _item_from_task(t, i)
        system = None
        if classifier is not None:
            item.task_type = classifier.classify(item.input).category
            if category_templates is not None:
                template_name = category_templates.get(
                    item.task_type, category_templates.get("_default", "default"))
                system = _system_prompt(template_name)
        remote = RemoteRunner(client, model=model, max_tokens=max_tokens, system=system)
        out = remote.run(item)
        score = scorers.score(out.answer, item)
        results.append(TaskResult(
            task_id=item.id, category=item.task_type, answer=out.answer, score=score,
            prompt_tokens=out.prompt_tokens, completion_tokens=out.completion_tokens,
            total_tokens=out.total_tokens, latency_s=None))
    return results


def rank_models_by_category(tasks: List[Dict[str, Any]],
                            per_model: Dict[str, List[TaskResult]],
                            accuracy_floor: float,
                            ) -> Dict[str, List[ModelCategoryRanking]]:
    # Categories come from the RECORDED results, not `tasks`' own (often
    # absent) category field -- when run_model_over_tasks classified each
    # task itself, `tasks` never had a category to begin with. `tasks` is
    # kept in the signature for stability; it's not read here anymore.
    del tasks
    categories = sorted({r.category for results in per_model.values() for r in results})
    ranking: Dict[str, List[ModelCategoryRanking]] = {}
    for cat in categories:
        rows = []
        for model, results in per_model.items():
            cat_results = [r for r in results if r.category == cat]
            if not cat_results:
                continue
            stats = evaluate_by_category(cat_results)[cat]
            rows.append(ModelCategoryRanking(
                category=cat, model=model, accuracy=stats.accuracy,
                total_tokens=stats.total_tokens, avg_tokens=stats.avg_tokens,
                clears_floor=stats.accuracy >= accuracy_floor))
        # floor-clearing models first, cheapest first; non-clearing models after (still visible)
        rows.sort(key=lambda r: (not r.clears_floor, r.total_tokens))
        ranking[cat] = rows
    return ranking


def run_bakeoff(tasks: List[Dict[str, Any]], models: List[str], client,
                accuracy_floor: float, max_tokens: int = 64,
                cached: Optional[Dict[str, List[TaskResult]]] = None,
                classifier: Optional[Classifier] = None,
                category_templates: Optional[Dict[str, str]] = None,
                ) -> Tuple[Dict[str, List[TaskResult]], Dict[str, List[ModelCategoryRanking]]]:
    """Run each allowed model over the task set ONCE (skipping any already in
    `cached` -- record-then-replay, D3), then rank per category. Ranking
    itself is pure arithmetic over already-paid-for records, same as the P1
    frontier tracer never re-pays. `classifier`/`category_templates` are
    forwarded to `run_model_over_tasks` -- see its docstring."""
    per_model: Dict[str, List[TaskResult]] = dict(cached) if cached else {}
    for model in models:
        if model not in per_model:
            per_model[model] = run_model_over_tasks(
                client, model, tasks, max_tokens,
                classifier=classifier, category_templates=category_templates)
    ranking = rank_models_by_category(tasks, per_model, accuracy_floor)
    return per_model, ranking


# ---------------------------------------------------------------------------
# Record-then-replay persistence (same pattern as schema.save_records, sized
# for TaskResult / a model->results map instead of the P1 Record schema).
# ---------------------------------------------------------------------------

def save_task_results(results: List[TaskResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def load_task_results(path: str) -> List[TaskResult]:
    with open(path) as f:
        raw = json.load(f)
    return [TaskResult(**r) for r in raw]


def save_bakeoff_records(per_model: Dict[str, List[TaskResult]], path: str) -> None:
    with open(path, "w") as f:
        json.dump({m: [asdict(r) for r in rs] for m, rs in per_model.items()}, f, indent=2)


def load_bakeoff_records(path: str) -> Dict[str, List[TaskResult]]:
    with open(path) as f:
        raw = json.load(f)
    return {m: [TaskResult(**r) for r in rs] for m, rs in raw.items()}

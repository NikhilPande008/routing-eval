# CONTEXT.md — Hybrid Token-Efficient Routing Agent (AMD ACT II, Track 1)

Canonical source of truth. Stable facts, resolved decisions, and structure only.
Rolling status lives in HANDOFF.md; the decision log lives in DECISIONS.md.
Change this file deliberately.

## Goal

Place top-3 on the Track 1 leaderboard. The score is **remote token count +
output accuracy** on a hidden standardized scoring environment. There is no
demo/product story in the score — this is a pure metric race.

Facts (verified from the lablab event page, 2026-07-01):
- Kickoff **July 6**, submission deadline **July 11** (~5-day build window).
- Models and tasks are revealed at kickoff, not before.
- Dev hardware is free (AMD MI300X dev cloud available); **local-model tokens
  count as zero** toward the score. Only remote tokens are metered.
- Hard submission gates: containerized app, public GitHub repo, README runnable
  on a clean environment.

Facts (official Track 1 participant guide, 2026-07-08 — see D17–D20):
- Local inference is permitted and **counts toward accuracy at zero token
  cost** — a local-only answer is graded, escalation is a pure cost lever.
- The accuracy gate is an **LLM-judge** external to us, not a scorer we design.
- Grading VM: **4 GB RAM / 2 vCPU**, **10-minute total** budget, **30s/request**
  budget.
- I/O contract: container reads **`/input/tasks.json`**, writes
  **`/output/results.json`** (`[{task_id, answer}, ...]`), exits 0. Implemented
  by `routing_eval/conformance.py` + `routing-eval score`.

## Architecture (resolved)

Three-tier cascade: `item → local attempt → gate → (keep local | escalate to
remote) → token accounting`.

- **Cascade, not a predictive pre-router.** Local tokens are free, so attempting
  local first is costless and yields richer gate signals (logprobs,
  self-consistency) than deciding before generation. A pre-router helps only as a
  latency optimization *if* the scoring env has a per-item time budget (unknown).
- **Constrained optimization.** Minimize remote tokens subject to
  accuracy ≥ floor. The floor is a constraint, not the objective. Operate just
  above the floor plus a safety margin sized to eval-set variance — the
  aggressive edge. Finding that edge requires the frontier, which is why the eval
  harness (P1) is priority one.
- **The gate is where the intelligence lives.** The harness is gate-agnostic; it
  consumes a `confidences` map and compares gates empirically on the same records.

## The P1 ↔ P2 contract

`routing_eval/schema.py::Record` is the single boundary. P2 produces records from
real models; the harness only consumes them. Key fields:
- `local_score`, `remote_score` — floats in [0,1] from the item's scorer.
- `remote_total_tokens` — what the competition counts if the item escalates.
- `confidences: {name -> float}` — one score per candidate gate.
  **Convention: higher = keep local.** Negate any native signal that points the
  other way (entropy, perplexity) before storing.

Record-then-replay: run local+remote on the whole dev set **once** (spends remote
tokens; dev tokens are free toward the score) → `records.json`. Sweeping the
threshold and comparing gates over that file is pure arithmetic and costs zero
tokens. Cache `records.json`; never re-pay.

## Repo layout

```
routing_eval/
  schema.py     Item + Record (the P1/P2 contract) + JSON load/save
  scorers.py    pluggable accuracy scorers (now a local proxy only -- real judge is external, D18)
  frontier.py   THE CORE: sweep + oracle + operating point + gate efficiency
  datasets.py   synthetic correct-by-construction stand-ins (dense borderline)
  mock.py       mock runners; validation instrument for testing without models
  runner.py     ModelRunner protocol (P2 boundary) + Fireworks call sketch
  llm/          OpenAICompatibleClient + StubClient, LocalRunner + RemoteRunner
  report.py     text report + CSV export + dependency-free ASCII plot
  taskio.py     shared task-dict helpers (load_tasks, task_id, task_prompt)
  modelids.py   ALLOWED_MODELS parsing + Fireworks model-ID normalization (D23)
  modelselect.py  Step 2: probe-local (local viability) + bakeoff (model ranking)
  classify.py   Step 3: cheap pluggable category classifier
  policy.py     Step 3: routing policy engine + draft-policy generator
  routing_policy.default.json  checked-in safe default (package data)
  conformance.py  scoring entrypoint: /input/tasks.json -> /output/results.json
  cli.py        `run`, `frontier`, `score`, `probe-local`, `bakeoff`, `generate-policy`
tests/          67 tests; frontier logic verified against a hand-worked case
skills/routing/ routing:run | routing:frontier | routing:verify | routing:end-session
```

Runtime dependencies: **none** (pure stdlib). `pytest` is dev-only. This is a
deliberate constraint so the container runs on a clean environment.

## Two-surface workflow

Split by concern; sync through files, never through a shared chat. Whoever picks
up a session reads HANDOFF.md first.

**Claude Code** owns the repo, compute, and API keys — therefore everything that
touches them: P2 and all routing/gate code, tests, the container, and *running*
the harness (`routing:run` needs the vLLM endpoint + Fireworks key). Also
τ-tuning, fine-tuning, and post-kickoff wiring of the revealed models/scorer.

**Cowork** owns everything that would otherwise steal context from the build
loop: intel (lablab + Discord watch for env spec, model list, token-count
definition, per-item time budget), submission collateral (cover image, slides,
~1-min video script — none affect the score), and the strategic overclaim-
policing pass. Cowork reasons over the harness's *outputs* (frontier reports,
CSVs) with zero code context.

## Handoff contract

Three files, in the repo:
- **CONTEXT.md** (this file) — canonical, stable.
- **HANDOFF.md** — rolling baton; engineering lane (Claude Code) + intel/strategy
  lane (Cowork). Read first, every session.
- **DECISIONS.md** — append-only; every resolved fork + a one-line why. Kills
  re-litigation and keeps both contexts small.

One rule (the Zero-Gemini-Logic analog): **`records.json` + the frontier report
is the only performance artifact either surface trusts.** A strategy claim not
backed by a frontier number does not enter DECISIONS.md.

## Known unknowns (resolved at kickoff → each becomes a DECISIONS entry)

1. ~~Scoring-env spec~~ → **resolved, D19**: 4GB RAM / 2 vCPU.
2. Token-count definition (prompt only? completion only? total?) → the meter to
   minimize. Current code minimizes total to be safe. **Still open.**
3. ~~Accuracy metric + threshold~~ → **resolved, D18**: LLM-judge, external.
   The operating-point floor itself is still open (depends on judge behavior).
4. ~~Per-item time budget~~ → **resolved, D19**: 30s/request, 10 min total. This
   bounds self-consistency to small N and makes a latency pre-router
   unnecessary at this budget (still cascade-first per D1).
5. Remote catalog: **resolved + confirmed live, D22 (2026-07-09)**. Real
   `ALLOWED_MODELS`: `gemma-4-31b-it`, `gemma-4-26b-a4b-it`,
   `gemma-4-31b-it-nvfp4`, `minimax-m3`, `kimi-k2p7-code`. `minimax-m3` and
   `kimi-k2p7-code` confirmed live-callable with real answers; the three
   Gemma variants require an on-demand deployment first and are deliberately
   not wired in yet (not blocking anything). Local model choice is still
   ours, bounded by #1 — **still open**.

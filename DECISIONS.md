# DECISIONS.md — append-only

Every resolved fork, with a one-line why. Newest at the bottom. Do not rewrite
history; supersede a decision with a new dated entry that references it. A
decision about performance does not go here unless a frontier number backs it.

---

### 2026-07-01 — Architecture & harness (P1)

- **D1 · Cascade over predictive pre-router.** Local tokens are free, so
  attempting local first is costless and yields richer gate signals than deciding
  before generation. A pre-router is a latency optimization only, viable only if
  the scoring env has a per-item time budget (unknown). → cascade dominates now.
- **D2 · Frame as constrained optimization.** Minimize remote tokens s.t.
  accuracy ≥ floor; operate at the aggressive edge (floor + margin sized to eval
  variance). The floor is a constraint, not the objective.
- **D3 · Record-then-replay architecture.** Decouple the expensive step (run
  local+remote once → records.json) from the free step (sweep threshold). Lets
  the frontier logic be fully tested with mock models now, and lets τ-tuning /
  gate comparison cost zero tokens later.
- **D4 · Zero runtime dependencies, pure stdlib (no numpy).** The container must
  run on a clean env; a zero-dep harness removes an entire class of failure.
- **D5 · Synthetic, correct-by-construction stand-ins with a dense borderline
  band.** Test the harness before real data exists; the borderline region is
  where the escalate/keep decision breaks and where happy-path testing lies.
- **D6 · Gate-agnostic harness; gate is a pluggable `confidences` map.**
  Convention: higher = keep local. Lets gates be compared empirically on the same
  records rather than chosen by intuition.
- **D7 · Oracle + gate-efficiency as the standing diagnostic.** Efficiency
  (oracle/gate ≤ 1) says which lever to pull: low → fix the confidence signal;
  even-oracle-infeasible → fix the model, not the gate.
- **D8 · records.json + frontier report is the only trusted performance
  artifact** (Zero-Gemini-Logic analog). No unbacked performance claim enters
  this log.

### 2026-07-01 — Workflow

- **D9 · Split surfaces by concern.** Claude Code owns repo/compute/keys → all
  code, tests, container, and running the harness. Cowork owns intel, submission
  collateral, and strategy-over-outputs. Sync through CONTEXT/HANDOFF/DECISIONS,
  never a shared chat.
- **D10 · Four repeatable procedures as skills:** `routing:run`,
  `routing:frontier`, `routing:verify`, `routing:end-session`. Encodes the loop
  so neither surface re-explains a procedure.

### 2026-07-01 — Routing scaffold (P2)

- **D11 · HTTP via stdlib `urllib`, not `requests`.** Preserves the zero-runtime-
  dependency property into P2. The vLLM server is external infra, not a Python dep
  of this package.
- **D12 · Stub client double for offline testing.** All runner / gate / router
  logic is tested without keys or network (25 tests total). Only a guarded live
  smoke (`scripts/live_smoke.py`) touches the real API, and it skips cleanly
  without credentials.
- **D13 · Gates emit "higher = keep local" confidence from the LOCAL output only
  (free).** An LLM-as-judge gate is deliberately excluded — it needs a second
  model call, so it's a billed feature, added only if the frontier shows it earns
  its tokens.
- **D14 · Implemented gate menu:** deterministic (structural veto), logprob
  (geometric-mean token prob), self-consistency (needs n>1 local samples; free
  here). Learned probe (FrugalGPT-g) included as a stdlib logistic regression,
  **fit post-reveal** on labeled dev records.
- **D15 · Router shares runners + gates with the record-builder,** so the τ
  calibrated on the frontier is the τ the agent uses. The real `build_records`
  emits the same P1 Record schema, so the frontier tracer sets τ on real data
  unchanged.

### 2026-07-01 — Container (P4)

- **D16 · Container = `python:3.12-slim`, installed via `pip install .` from
  pyproject, non-root, zero runtime deps.** The image footprint is base Python +
  this package; build pulls only the build backend. Default entrypoint is the eval
  CLI; the scoring-time entrypoint is wired at kickoff once the scoring harness
  interface is known. `scripts/container_smoke.sh` is the clean-env gate.
  Note: the clean `pip install .` + entrypoint + quickstart path is proven; the
  `docker build` itself runs on a machine with a Docker daemon (not the dev
  sandbox), so the container gate is closed by running the smoke script there.

### 2026-07-08 — Official Track 1 participant guide (post-kickoff correction)

> Note: this entry was requested as "reverses D22," but this log has no D22 —
> entries run D1–D16 only. Not fabricating a history that doesn't exist here;
> if D22 lives in a Cowork-side note that never synced to this file, reconcile
> it against the facts below rather than the number.

- **D17 · Local inference is permitted and counts toward accuracy, at zero
  token cost.** The grading VM scores a local-only answer directly — escalating
  to remote is purely a token-cost lever, not a prerequisite for the answer to
  count. Extends the existing "local tokens = 0 toward the score" premise
  (CONTEXT.md) to confirm local answers are graded, not just free to attempt.
- **D18 · The accuracy gate is an LLM-judge**, not a scorer we choose.
  Resolves Known Unknown #3 (accuracy metric + threshold). `scorers.py`'s job
  changes from "pick the right metric" to "produce answers in the shape a
  judge model can grade" — the rubric/judge itself is external, not ours to
  design. Closes the D-open-fork "Accuracy metric + threshold."
- **D19 · Grading VM: 4 GB RAM / 2 vCPU, 10-minute total budget, 30s/request
  budget.** Resolves Known Unknown #1 (scoring-env spec) and #4 (per-item time
  budget). Bounds local model size/quantization to what fits 4GB/2vCPU, and
  bounds self-consistency (N-sample) to whatever N fits in 30s/request.
- **D20 · I/O contract: container reads `/input/tasks.json`, writes
  `/output/results.json`** (`[{task_id, answer}, ...]`), exits 0. Resolves the
  "scoring-time entrypoint wired at kickoff" placeholder in D16. `routing-eval
  score` (added this session) implements this; see [conformance.py].

### 2026-07-08 — Fine-tuning tutorial: open conflict with D18 (NOT resolved)

- **D21 · OPEN CONFLICT, not a resolution — do not reverse D18 or D19 yet.**
  The official Track 1 fine-tuning tutorial
  ([lablab](https://lablab.ai/ai-tutorials/fine-tune-llm-query-router-amd),
  [repo](https://github.com/Stephen-Kimoi/fine-tune-llm-query-router-amd)) has
  the fine-tuned local model output ONLY a routing label, never the final
  answer — every task still gets a real Fireworks call, and the tutorial
  states explicitly the fine-tuned piece "can only be the routing decision,
  never the answer itself." That directly contradicts the participant guide
  line D18 extends ("local model inference inside the container is permitted
  and counts toward accuracy, but not toward the token score") — i.e. whether
  a local-only answer can ever be scored at all is now unconfirmed and
  possibly wrong. We've asked Discord; no answer yet. **Until resolved:**
  probe-local's (Step 2) "local answers can be scored" premise and
  `policy.py`'s local-viable routing tier (Step 3) rest on an unconfirmed
  premise — see HANDOFF.md blocker. Nothing in `modelselect.py`, `policy.py`,
  or `classify.py` changed this session; this is a documentation checkpoint so
  work doesn't keep building on the unconfirmed premise before Discord answers.
- **D21a · Corroborated, independent of the conflict above: the
  classifier-only-local pattern is validated.** A local model choosing WHICH
  Fireworks model/category to route to — never producing the final answer
  itself — costs zero tokens and is exactly what the tutorial demonstrates and
  exactly what `classify.py` already does. This part is safe under EITHER
  reading of the D21 conflict; no change needed.
- **D21b · Supporting data point for bake-off, not a reason to change it.**
  The tutorial reports a single cheap model (`gpt-oss-120b`) matched the
  expensive model's accuracy on their eval set with fewer total tokens than a
  router. This is exactly the question Step 2's bake-off (`routing_eval.
  modelselect`) is already built to answer empirically on the real
  `ALLOWED_MODELS` catalog — it corroborates the bake-off's premise, it
  doesn't change the bake-off's design (D8: no performance claim without our
  own frontier number). Their README also names the underlying reason: their
  labeled dataset is heavily skewed (80 easy / 3 hard of 83 queries) even
  after an adversarial batch — for THEIR model pair there's barely an
  accuracy gap to route around. Doesn't tell us anything about our eventual
  `ALLOWED_MODELS` pair; still zero tokens spent confirming or refuting it.
- **D21c (still open, code-level corroboration read directly, 2026-07-08) ·
  The tutorial's actual code has no path that ever uses a local-only answer.**
  Read `agent.py` directly (not just the README): `main()` unconditionally
  calls `chat(model, task["prompt"], ...)` — a real Fireworks call — for
  EVERY task regardless of the routing decision; `route()` only ever selects
  WHICH model to call, never skips the call. There is no `ROUTER_MODE` that
  returns a local answer as the final answer. This is stronger evidence than
  the README prose alone (D21) — a working reference implementation built
  for this same competition, not just tutorial text — but it's still the
  tutorial author's interpretation, not a quote of the actual rule text, and
  still doesn't override D18/D19 without Discord confirmation. Two additional
  data points surfaced while reading this code, both flagged as second-hand
  and NOT acted on yet:
  - `agent.py`'s sample task shape uses `task["prompt"]` — consistent with
    (not proof of) `taskio.py`'s existing first-choice guess in
    `_PROMPT_FIELDS`. No code change; this was already our default.
  - The tutorial repo's disclaimer states, citing "the official Participant
    FAQ for AMD Developer Hackathon: ACT II," that the allowed Fireworks
    models are the **MiniMax and Kimi K series** (their own code uses
    `gpt-oss-120b`/`deepseek-v4-pro`/`glm-5p2` as stand-ins, explicitly NOT
    the real allowed models). This is a lead on Known Unknown #5, not a
    resolution — it's second-hand (their citation of the FAQ, not the FAQ
    itself) and we have not independently verified it. See CONTEXT.md Known
    Unknown #5 and HANDOFF.md.
  - `fireworks_client.py`'s docstring says token usage is tracked in one
    Fireworks-wrapper function "matching how the hackathon's judging proxy
    records tokens centrally" — implies token accounting happens server-side
    on the platform, not something we self-report. Doesn't resolve Known
    Unknown #2 (prompt/completion/total) either way; noted, not acted on.

### 2026-07-09 — Real models, real practice tasks (Step 0 + conformance dry run)

- **D22 · `ALLOWED_MODELS` is confirmed to include `minimax-m3` and
  `kimi-k2p7-code`, both live-callable, at zero cost to confirm.** This
  upgrades D21c's tutorial-sourced "lead" (second-hand, unverified) to a
  first-hand fact: read the real `ALLOWED_MODELS` env var directly, then made
  real HTTP calls to both models against the real Fireworks endpoint and got
  real answers back (e.g. correctly computed "144" for practice-02's math
  problem). Real `ALLOWED_MODELS` (2026-07-09): `gemma-4-31b-it`,
  `gemma-4-26b-a4b-it`, `gemma-4-31b-it-nvfp4`, `minimax-m3`,
  `kimi-k2p7-code`. The three Gemma variants require an on-demand deployment
  first (per user instruction) — deliberately NOT wired in anywhere, and
  nothing in this session depends on them being available.
- **D23 · Fireworks model IDs require the full `accounts/fireworks/models/`
  path; `ALLOWED_MODELS` gives bare names — this was a real, load-bearing bug,
  not a hypothetical.** The very first live call (`minimax-m3` used as-is)
  404'd: `"Model not found, inaccessible, and/or not deployed"`. Confirmed via
  direct `curl` that the SAME model, prefixed
  (`accounts/fireworks/models/minimax-m3`), returns 200. Fixed by extracting
  `split_models`/`normalize_model_id` into a new shared `routing_eval/
  modelids.py` (avoids a conformance.py ↔ policy.py import cycle, same
  pattern as the earlier `taskio.py` extraction) and applying normalization
  at both entry points: `split_models()` (covers everything sourced from
  `ALLOWED_MODELS` — `score`, bake-off, policy fallback) and `policy.py`'s
  `_resolve_model()` (covers a hand-authored `routing_policy.json` "model"
  field, so a human writing a short name doesn't silently 404 at scoring
  time). Idempotent — already-prefixed strings pass through unchanged, so
  this is safe regardless of what shape the real launch-day `ALLOWED_MODELS`
  turns out to be.
- **D24 · Real practice-task schema confirmed: `{task_id, prompt}` only —
  no category, no gold, no scorer.** Matches `taskio.py`'s existing
  `_PROMPT_FIELDS` first-choice guess exactly; no code change needed there.
  `scripts/fixtures/practice_tasks.json` (renamed from
  `placeholder_tasks.json`) now holds the real 8 tasks from the guide. Since
  the real tasks carry no category or gold, `probe-local`/`bakeoff`/
  `generate-policy` run against this file will see every task as
  `category="uncategorized"` with no gold to score against — that was always
  true of the real contract, not a new limitation; see
  `scripts/fixtures/README.md`.
- **D25 · Conformance dry run (Step 1 + Step 3) proven end-to-end against
  real infrastructure — raw data only, not a performance claim (D8).** Real
  `linux/amd64` submission image built: **43.4MB** (well under the 10GB
  cap). Ran via `scripts/conformance_smoke.sh` (`REAL_FIREWORKS=1` mode,
  added this session) against the real Fireworks endpoint, the real 8
  practice tasks, and the checked-in safe-default policy (`model: null` →
  resolved to `minimax-m3`, the first of the two confirmed-ready models).
  Wall-clock: **14s** (budget: 600s / 10 min, D19). Exit 0, valid
  `results.json`, container output byte-identical to the local dry run.
  **Total tokens across all 8 tasks: 1,370** (per-task: 150, 152, 147, 213,
  176, 182, 165, 185) — first real baseline number, zero token optimization
  applied (every task still routes to Fireworks; no local-viable tier is
  wired, per the still-open D21 conflict). Answers were spot-checked and are
  substantively correct (e.g. practice-02's math, practice-07's logic puzzle)
  but accuracy is NOT graded by us — the real judge (D18) is external, so no
  accuracy claim is made here, only token cost.

### 2026-07-09 — D21 RESOLVED: local-only answers are scored (Discord)

- **D26 · D21 is resolved, in favor of D17/D18's ORIGINAL reading — not the
  fine-tuning tutorial's more conservative one.** Discord answer (organizer/
  mod, 2026-07-09), quoted in full since it's now load-bearing: "A local
  model's output can contribute to the final answer, as local model
  inference inside the container is permitted and counts toward accuracy.
  However, all Fireworks API calls must go through FIREWORKS_BASE_URL for
  token efficiency, meaning that while local models can provide answers,
  they won't contribute to the token score. So, local models can be used for
  actual submissions, but their outputs won't affect the token count." This
  confirms D17 exactly: a local-only answer is graded, and costs zero
  tokens. The fine-tuning tutorial's "every task gets a real Fireworks call"
  pattern (D21) was one author's implementation choice, not evidence of a
  platform constraint — nothing stopped them from using local-only answers,
  they simply didn't build that path.
  **Consequence, superseding the D21 caution:**
  - `probe-local`'s (Step 2) local-viability numbers are now safe to rely on
    for real — the premise they were built on ("local answers can be
    scored") is confirmed, not just architected-in-case.
  - `policy.py`'s `"local"` tier (Step 3) is now safe to wire into a real
    submission policy — previously it was correctly-built but
    correctly-unused pending this answer.
  - D21a (classifier-only-local is separately validated) and D21b (bake-off
    is the right instrument for model selection) are unaffected either way
    and remain true.
  - Not yet done: no policy has actually been generated or switched to use
    the local tier. This decision clears the way; it doesn't do the work.
    See HANDOFF.md Next.

### 2026-07-09 — Category-specific prompt templates (format fix, not a model
### competence issue)

- **D27 · Fixed a real answer-format defect in 3 of the 8 real practice
  tasks — root cause was `DEFAULT_SYSTEM`'s "briefly, do not explain"
  instruction, not model competence.** The D25 dry run's raw answers for
  practice-06/practice-08 (code_debug/code_gen) were markdown-fenced code
  blocks (` ```python ... ``` `) rather than bare runnable code, and
  practice-03 (sentiment) returned only `"Mixed"` with no justification,
  though the guide expects a label AND a justification. **Correction to the
  record**: last session's own summary table paraphrased the two code
  answers as prose descriptions (e.g. "dedupes via set, sorts, returns
  second element"), dropping the `reverse=True` detail — that paraphrase,
  not the model, is what made the returned algorithm look wrong. Verified by
  directly executing the actual returned code (not eyeballing, per
  instruction): both the old and new `practice-08` code correctly compute
  the second-largest value for `[1,2,3,4,5,5]` → `4` and several other
  duplicate-containing cases. The fencing and missing-justification issues
  were real, though.
  **Fix, kept in per-category config, nothing hardcoded per-task:**
  - `policy.py` `PROMPT_TEMPLATES` gained `code_only` (output only runnable
    code, no fences, no prose) and `sentiment_with_justification` (label +
    one-line justification, exact form specified).
  - `routing_policy.default.json` (the checked-in **safe default**) now maps
    `code_debug`/`code_gen` → `code_only` and `sentiment` → 
    `sentiment_with_justification`. `tier`/`model` are UNCHANGED for every
    category (still `fireworks`/`null`) — only `prompt_template` differs.
    The safe default's core property (routes everything to Fireworks) is
    untouched.
  - Added `_strip_code_fence()` as a deterministic backstop in
    `PolicyRouter._call_fireworks` for the `code_only` template — the
    instruction alone doesn't guarantee compliance, so a fenced answer is
    still stripped to bare code even if the model fences it anyway.
    Idempotent/harmless on unfenced text.
  - `classify.py`'s `DEFAULT_KEYWORDS` gained `sentiment`/`code_debug`/
    `code_gen` keyword sets, calibrated so these 3 practice tasks actually
    reach the new templates (a correct template nobody routes to fixes
    nothing). Verified live: re-ran the real conformance dry run
    (`REAL_FIREWORKS=1`, real 8 tasks, real Fireworks, both locally and
    through a fresh container build) — practice-03 now returns
    `"Mixed: The review praises the battery life but criticizes the
    screen's durability."`, practice-06/08 now return bare unfenced code,
    both re-verified correct by execution against multiple cases including
    duplicates.
  - Token cost moved **1,370 → 1,537** across the 8 tasks (longer system
    instructions + a fuller justification + more defensive code with error
    handling). An honest, expected tradeoff for correct format, not a
    regression — not a performance claim either way (D8).
- **D28 · Step 2: classifier expanded from 3 to all 8 real categories.**
  Added `math` (practice-02), `summarization` (practice-04),
  `entity_extraction` (practice-05), `logic` (practice-07); `knowledge`
  (practice-01) was already correct and left as-is. Every keyword was
  chosen because it is a literal token present in exactly one of the 8 real
  prompts and absent from the other 7 — verified by classifying all 8 real
  prompts, not assumed (one tie surfaced and was fixed: `summarize` alone
  tied 1-1 against `knowledge`'s `"city"` hit on practice-04; added
  `"sentence"` as a second summarization signal to break it cleanly).
  `wordplay` still has no matching practice task and remains untested
  against real data. This is calibration against these 8 known examples,
  not a general-purpose taxonomy — expect misses on dissimilar tasks.
  Purely a classification-accuracy change: `routing_policy.default.json` has
  no entries for `math`/`summarization`/`entity_extraction`/`logic`/
  `knowledge`, so all 5 still correctly fall through to `"_default"` (the
  generic template) — no answer-format behavior changed for them.

### 2026-07-09 — Classifier generalization test: fails badly on paraphrases

- **D29 · The keyword classifier does not generalize past the 8 literal
  calibration prompts — 3/36 (8%) hit rate on hand-written paraphrases.**
  D28 calibrated `DEFAULT_KEYWORDS` by finding a literal token unique to
  each of the 8 real practice prompts; that was always calibration-to-8-
  examples, not a taxonomy, and this session tested exactly that limit. 36
  hand-written variants (4 per category, including `wordplay` which has no
  practice-task equivalent — 8 real categories × 4 = 32, plus 4 bonus
  `wordplay` cases), each avoiding its category's trigger keyword where
  avoidable (e.g. "condense this to one line" instead of "summarize"). Per
  category: `knowledge` 2/4, `math` 1/4, everything else 0/4. Report-only,
  per instruction — nothing fixed. Two distinct failure modes, both
  structural, not incidental:
  1. **Silent non-match → `"uncategorized"`.** Most misses: a paraphrase
     that genuinely avoids every keyword gets zero signal at all, not a
     low-confidence guess. Majority of the 33 misclassifications.
  2. **Wrong-category collision via a generic word.** `"what"`/`"who"`/
     `"where"` (knowledge) and `"how"`/`"product"` (math) are common enough
     in ordinary phrasing that they hijack classification for unrelated
     categories — e.g. a math word problem containing "what quantity
     remains" classified `knowledge`; a summarization prompt mentioning "a
     new product launch" classified `math`; a code_debug prompt asking to
     "reverse a string" classified `wordplay`. This reproduced live even
     while writing this session's own new unit tests — a test prompt
     "What is the sentiment here?" for a `sentiment` fixture classified
     `knowledge` via `"what"`, forcing a rewrite to
     `"Classify the sentiment of this comment."` The same failure mode a
     real, unseen paraphrase would hit.
  Practical read: `DEFAULT_KEYWORDS` is safe for exactly the 8 practice
  prompts (that's what it was built and verified against, D28) but should
  not be assumed to work on the real graded task set once it's revealed,
  even if that set is thematically similar. Script + fixture kept for
  reuse: `scripts/classifier_paraphrase_check.py` +
  `scripts/fixtures/classifier_paraphrases.json`.
- **D30 · Real bake-off (Step 3): `kimi-k2p7-code` beats `minimax-m3` on
  tokens in EVERY category tested, by roughly 2x — first real signal,
  n=1/category, not a general claim.** Ran both confirmed-live models
  (D22) over all 8 real practice tasks, through the now-fixed per-category
  prompt templates (D27/D28 — classification wired into the bake-off itself
  this session, see below), record-then-replay to
  `/tmp/real_bakeoff_records.json`. `accuracy=0.00`/`clears_floor=no`
  everywhere is a known artifact, not a finding: the real practice tasks
  carry no gold answer (D24), so `scorers.score()` always returns 0 — the
  floor-clearing column from `rank_models_by_category` is meaningless here
  and was ignored in favor of raw token cost, which IS real. Per-category
  tokens (kimi / minimax): code_debug 142/216, code_gen 134/272,
  entity_extraction 72/176, knowledge 54/150, logic 74/165, math 57/152,
  sentiment 111/193, summarization 113/213. Totals: **kimi 757, minimax
  1,537** — kimi wins every category, no split observed. minimax's total
  matches the D27 conformance-dry-run total exactly (1,537), a consistency
  check that both runs hit the same real endpoint the same way. kimi's
  answers were spot-checked and its two code answers (practice-06/08)
  re-verified correct by execution, same as D27 — cheaper is not
  cheaper-because-truncated here. **Not a general claim**: 1 sample per
  category, no statistical basis, and D18's real judge might disagree with
  our own scoring proxy entirely.
  **Code change enabling this** (not just a one-off script): `bakeoff` was
  previously blind to categories on real tasks — `_item_from_task` read
  `task.get("category", "uncategorized")`, and real tasks have no such
  field, so every prior bake-off run silently collapsed into one
  `"uncategorized"` bucket. Fixed by threading an optional `classifier` +
  `category_templates` through `run_model_over_tasks`/`run_bakeoff`
  (`modelselect.py`) — when given, each task is classified the same way the
  real `score` path would, and gets the SAME per-category system prompt
  (`code_only`/`sentiment_with_justification`/`default`) a real submission
  would use, so the bake-off is now apples-to-apples with reality instead
  of comparing models under a prompt nobody actually ships. Also fixed
  `rank_models_by_category` to derive categories from the recorded
  `TaskResult`s instead of the (often absent) task-dict field — the old
  version would have silently produced a single `"uncategorized"` ranking
  bucket even with a classifier wired in upstream. New shared
  `routing_eval/prompts.py` holds `PROMPT_TEMPLATES` (moved out of
  `policy.py`, re-exported there for compatibility) because `modelselect.py`
  can't import `policy.py` — `policy.py` already depends on `modelselect.py`
  for `LocalViability`/`ModelCategoryRanking`. Same cycle-avoidance pattern
  as `taskio.py`/`modelids.py`. CLI: `routing-eval bakeoff` now classifies
  and applies category templates by default (`--no-classify` to opt out,
  `--policy` to point at a specific `routing_policy.json` for templates).

### 2026-07-09 — First real submission dry run: hardened + kimi-pinned

- **D31 · Hardening: retry-on-opposite-model before empty is the last
  resort, and D19's 30s/request timeout is now enforced on Fireworks
  calls (it previously wasn't).** Two independent fixes, both proven by
  the current rank-1 submission's pattern (omerdduran/token-router):
  1. A blank or failed Fireworks answer now retries ONCE with a different
     model drawn from `ALLOWED_MODELS` (never a hardcoded model ID — see
     `PolicyRouter._pick_retry_model`) before degrading to `""`. Empty is
     the last resort, not the first fallback. Verified live with a
     StubClient: blank-then-real, exception-then-real, and both-blank (only
     then does it degrade to empty, and only after the retry actually
     fired — proven via the log line, not assumed).
  2. `RemoteRunner.run()` gained a `timeout` param (it had none before,
     unlike `LocalRunner.run()` which already did) and `PolicyRouter` now
     passes `entry.timeout_s or default_timeout_s` (D19's 30s) to every
     Fireworks call. Matters more now that a task can make 2 calls: without
     this, 2 uncapped attempts (the client's 60s default each) could reach
     120s for one task — 8 such tasks would blow the 600s/10-min budget.
     With the fix, worst case is 2×30s=60s per task, 480s for all 8 -- safely
     inside budget with margin.
  - Confirmed `reasoning_effort="none"` is already sent on every Fireworks
    call — no code change needed. Verified by capturing the actual raw HTTP
    request body from a real live call (not just reading the code): the
    JSON body sent to Fireworks includes `"reasoning_effort": "none"`, set
    by `RemoteRunner`'s existing default and never overridden by
    `PolicyRouter`.
- **D32 · Checked-in default policy re-pinned from `model: null` to
  `model: "kimi-k2p7-code"` for every category — an informed choice backed
  by D30's real bake-off, not the old "safe null" placeholder.** All 4
  entries (`code_debug`, `code_gen`, `sentiment`, `_default`) now pin
  `kimi-k2p7-code`; `tier` stays `fireworks` everywhere (still no local
  tier wired — that's still a separate, deliberate step). D31's retry logic
  is the safety net if kimi is ever unavailable at grading time: the
  pinned model isn't a single point of failure anymore, `ALLOWED_MODELS`'
  other entries are.
- **D33 · Submission dry run (hardened + kimi-pinned), proven both locally
  and through a fresh container build against real Fireworks — not
  submitted, per instruction.** Real 8 practice tasks, real
  `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` scoped to `minimax-m3,
  kimi-k2p7-code` (Gemma still excluded, still not on the critical path).
  Every task resolved via `kimi-k2p7-code` on the first attempt — no retry
  needed, so D31's retry path wasn't exercised live this session (only
  proven via StubClient tests, D31 above). **Total: 763 tokens** across all
  8 tasks (vs. 757 in D30's bake-off using the generic-only prompt
  comparison — consistent order of magnitude; small delta from longer
  category-specific instructions on some tasks). Both code answers
  (practice-06/08) re-verified correct by execution against
  `[1,2,3,4,5,5]` → `4` and other cases, same rigor as D27/D30. Container:
  built `linux/amd64` via `docker buildx`, **178MB** (well under the 10GB
  cap — note this run reported a different size than D25's 43.4MB build;
  both are trivially under the cap, the delta is base-layer caching state
  after a Docker Desktop restart mid-session, not a code change).
  Wall-clock **11s** (budget 600s). `results.json` schema validated
  programmatically (not eyeballed) on both the local and container output:
  exactly `[{task_id, answer}, ...]`, all 8 `practice-01`..`08` echoed, no
  extra/missing keys. Container output differs from the local run only in
  minor LLM-level wording (e.g. "GPE" vs "LOCATION" for the same entity) —
  expected non-determinism, not a correctness or schema issue.
  **Registry reference NOT reported as a real value** — this repo has no
  git remote configured yet (not pushed to GitHub), so no GHCR reference
  exists to report. Gave the template
  (`ghcr.io/<owner>/<repo>:<tag>`) instead of fabricating one.
  **Infra note**: Docker Desktop hung completely mid-build this session
  (daemon unresponsive to `docker info` for 10+ minutes, unrelated to any
  code here) — required a full kill and relaunch to recover. Not a
  routing-eval issue, flagging in case it recurs close to the deadline.
  **Not submitted to the hackathon platform** — that's the user's call
  after eyeballing the 8 answers above.

---

## Open forks (resolve at kickoff; add a dated entry each)

- Gate choice: (A) deterministic checks, (B) logprob/perplexity threshold,
  (C) self-consistency/agreement, (D) learned probe (FrugalGPT-g), (E) LLM-judge
  as a feature. Plan: layer A → B/C → optional D; let the frontier pick the
  winner on the real task.
- Local model + quantization, now bounded by D19 (4GB RAM / 2 vCPU / 30s per
  request) but not yet picked.
- Token-count definition → which meter to minimize (code currently minimizes
  total).
- Whether to fine-tune (only if the oracle says you are capability-limited).

Resolved this session: ~~Accuracy metric + threshold~~ → D18 (LLM-judge,
external to us).

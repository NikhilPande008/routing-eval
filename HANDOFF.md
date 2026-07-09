# HANDOFF.md — rolling baton

Read this first. Update it at the end of every session (`routing:end-session`
does this). Two lanes: **ENG** (Claude Code writes) and **INTEL/STRATEGY**
(Cowork writes). Keep it short — details go in CONTEXT.md / DECISIONS.md.

---

Last updated: 2026-07-09 · by: first real submission dry run (hardened + kimi-pinned)

## State

- **✅ READY TO SUBMIT, pending your eyeball — nothing has been submitted
  (D31/D32/D33, 2026-07-09).** Two hardening fixes landed and were proven,
  then the checked-in default was re-pinned to the bake-off winner, then a
  full submission dry run ran clean both locally and through a fresh
  container build against real Fireworks.
  - **Hardening 1 — retry-on-opposite-model, empty is the last resort now,
    not the first fallback.** A blank/failed Fireworks answer retries once
    with a different `ALLOWED_MODELS` entry before degrading to `""`. Never
    a hardcoded model ID — picks from whatever's actually injected at
    runtime. Proven with 5 new tests (blank→retry→real answer,
    exception→retry→real answer, both-blank→empty only after a real retry
    attempt, single-model→no retry possible, and a timeout-bound check).
  - **Hardening 1b — D19's 30s/request timeout is now actually enforced on
    Fireworks calls.** `RemoteRunner.run()` had no `timeout` param before
    (unlike `LocalRunner`, which did) — the client's 60s default applied
    instead. Fixed, and it matters more now that a task can make 2 calls:
    without the cap, worst case was 2×60s=120s/task, 960s for 8 tasks — over
    budget. Now: 2×30s=60s/task, 480s worst case, safely inside 600s.
  - **Hardening 2 — `reasoning_effort=none` confirmed already correct, no
    code change.** Verified by capturing the actual raw JSON body of a real
    live Fireworks call, not just reading the code — `"reasoning_effort":
    "none"` is there, set by `RemoteRunner`'s existing default.
  - **Checked-in default re-pinned**: every category (`code_debug`,
    `code_gen`, `sentiment`, `_default`) now routes to `kimi-k2p7-code`
    explicitly (was `model: null`) — an informed choice backed by D30's real
    bake-off, not the old safe-null placeholder. Hardening 1 is the new
    safety net if kimi ever isn't available at grading time.
  - **Submission dry run, hardened + kimi-pinned, real infra throughout**:
    all 8 real practice tasks, real Fireworks, both a local run and a fresh
    `linux/amd64` container build. Every task resolved on kimi on the first
    attempt (no retry needed this run — the retry path is proven via tests,
    not exercised live here). **Total: 763 tokens** for all 8 tasks (down
    from 1,537 pre-bake-off, D27's baseline — roughly half). Both code
    answers re-verified correct by execution (not eyeballed), same rigor as
    every prior session. `results.json` schema validated programmatically:
    exactly `{task_id, answer}`, all 8 IDs echoed. Container: 178MB (well
    under 10GB), wall-clock 11s (budget 600s), exit 0.
  - **All 8 final answers** (container run, `/tmp/real_practice_results.json`
    on this machine):

    | task | answer |
    |---|---|
    | practice-01 | Canberra; Lake Burley Griffin. |
    | practice-02 | 144 |
    | practice-03 | Mixed: praises battery life, criticizes screen — both positive and negative sentiment |
    | practice-04 | (one-sentence budget summary, no fence/prose issues) |
    | practice-05 | Maria Sanchez-PERSON, Fireworks AI-ORGANIZATION, Berlin-LOCATION, last March-DATE |
    | practice-06 | bare `get_max` code, re-verified correct by execution |
    | practice-07 | Sam owns the cat. |
    | practice-08 | bare `second_largest` code, re-verified correct incl. `[1,2,3,4,5,5]`→`4` |

  - **No registry reference reported as real** — this repo has no git
    remote yet (not pushed to GitHub), so no GHCR image exists. Gave the
    `ghcr.io/<owner>/<repo>:<tag>` template instead of fabricating one; ask
    if you want exact push commands once you know the target repo.
  - **Infra hiccup, not a code issue**: Docker Desktop hung completely
    mid-session (unresponsive to `docker info` for 10+ minutes) and needed
    a full kill+relaunch to recover. Unrelated to routing-eval; flagging in
    case it recurs close to the deadline — the recovery playbook is:
    `pkill -f "Docker Desktop"` + `pkill -f com.docker`, then
    `open -a Docker`, then poll `docker info` (can take a couple minutes for
    a full cold VM boot after a forced kill).
  - **82 tests pass (was 77 at the start of this session: +5 for the retry
    logic). `routing:verify`: PASS** — tests + lint confirmed before AND
    after the Docker outage, both container gates (stub `routing:verify`
    gate and the real-Fireworks submission smoke) confirmed exit 0 after
    Docker recovered. Lint: same 2 pre-existing findings, no new ones.
- **⚠ D29 (2026-07-09): the classifier does NOT generalize — 3/36 (8%) hit
  rate on hand-written paraphrases of the 8 practice tasks.** Per-category:
  `knowledge` 2/4, `math` 1/4, all 6 others 0/4. This was always calibration
  against 8 literal examples (D28), and this session tested exactly that
  limit: 36 hand-written variants (4 per category × 8 real categories, plus
  4 bonus `wordplay` cases), each avoiding its trigger keyword where
  avoidable. Two failure modes, both structural: (1) a paraphrase that
  avoids every keyword gets `"uncategorized"`, no partial credit — most
  misses; (2) generic words (`"what"`/`"who"`/`"how"`/`"product"`) hijack
  classification into an unrelated category — reproduced live even while
  writing this session's own new unit tests (a `sentiment` test prompt
  "What is the sentiment here?" classified `knowledge` via `"what"`, had to
  be rewritten). **Practical read: do not assume `DEFAULT_KEYWORDS` works on
  the real graded task set, even if it looks similar to the 8 practice
  tasks.** Report only, nothing fixed, per instruction. Reusable:
  `scripts/classifier_paraphrase_check.py` +
  `scripts/fixtures/classifier_paraphrases.json`. Full failure-mode
  breakdown in DECISIONS.md D29.
- **✅ D30 (2026-07-09): real bake-off run — `kimi-k2p7-code` beats
  `minimax-m3` on tokens in EVERY category, ~2x cheaper overall (757 vs
  1,537 tokens). First real signal, n=1/category — not a general claim.**

  | category | kimi-k2p7-code | minimax-m3 |
  |---|---:|---:|
  | code_debug | 142 | 216 |
  | code_gen | 134 | 272 |
  | entity_extraction | 72 | 176 |
  | knowledge | 54 | 150 |
  | logic | 74 | 165 |
  | math | 57 | 152 |
  | sentiment | 111 | 193 |
  | summarization | 113 | 213 |
  | **total** | **757** | **1,537** |

  `accuracy`/`clears_floor` were 0.00/no for everything — a known artifact
  (real tasks carry no gold, D24), not a finding; ignored in favor of raw
  tokens. kimi's answers spot-checked and its two code answers re-verified
  correct by execution (same rigor as D27) — cheap here isn't
  cheap-because-truncated. No category split observed in this sample; one
  model won everywhere. minimax's total exactly matches the D27 dry-run
  total (1,537) — a consistency check that both hit the real endpoint the
  same way.
  **Code change enabling this**: the bake-off was previously blind to
  categories on real tasks (no `category` field on them, so everything
  silently became `"uncategorized"`) and used the generic prompt for every
  call regardless of category — not apples-to-apples with what `score`
  actually sends. `run_model_over_tasks`/`run_bakeoff` (`modelselect.py`)
  now take optional `classifier`/`category_templates` params; the CLI passes
  both by default (`--no-classify` to opt out). New `routing_eval/
  prompts.py` holds the prompt-template dict (moved out of `policy.py`,
  re-exported there) so `modelselect.py` can use it without an import cycle
  (`policy.py` already depends on `modelselect.py`) — same pattern as
  `taskio.py`/`modelids.py`. Records: `/tmp/real_bakeoff_records.json`.
- **✅ D27/D28 (2026-07-09): fixed a real answer-format defect in 3 of the 8
  practice tasks, then calibrated the classifier against all 8.** Root cause
  was the generic system prompt, not model competence — see below and
  DECISIONS.md D27/D28. **Correcting my own prior record**: last session's
  summary table paraphrased practice-06/08's code answers into prose and
  dropped a `reverse=True` detail, which is what made the algorithm look
  wrong — the actual code, re-verified by execution (not eyeballing), was
  correct both times. The real, confirmed issues were markdown-fenced code
  (not bare-runnable) and a sentiment label with no justification.
  - `policy.py` gained `code_only` and `sentiment_with_justification`
    prompt templates, wired into the checked-in safe default for
    `code_debug`/`code_gen`/`sentiment` (tier/model unchanged — still
    `fireworks`/`null` for every category). A deterministic
    `_strip_code_fence()` backstop guarantees bare code even if a model
    fences anyway.
  - `classify.py`'s keywords expanded from 3 fabricated categories to all 8
    real ones, each calibrated against a literal token in one specific real
    practice-task prompt (verified, not assumed) — `knowledge`, `math`,
    `sentiment`, `summarization`, `entity_extraction`, `code_debug`,
    `logic`, `code_gen`. `wordplay` remains untested (no matching practice
    task).
  - Re-verified live (`REAL_FIREWORKS=1`, real 8 tasks, real Fireworks,
    local run + fresh container build, byte-identical output): practice-03
    now returns a label + justification; practice-06/08 now return bare
    unfenced code, re-confirmed correct by executing the actual returned
    code against `[1,2,3,4,5,5]` and other duplicate-containing cases.
  - **Token total moved 1,370 → 1,537** across the 8 tasks — longer
    instructions + fuller answers, an expected tradeoff for correct format,
    not a regression or a performance claim (D8).
- **✅ D21 RESOLVED (D26, Discord organizer answer, 2026-07-09): local-only
  answers ARE scored, at zero token cost.** Confirms D17/D18's original
  reading — the fine-tuning tutorial's "always call Fireworks" pattern was
  one author's implementation choice, not a platform rule. **This unblocks
  `probe-local`'s local-viability numbers (Step 2) and `policy.py`'s
  `"local"` tier (Step 3) for real use** — previously correctly-built but
  correctly-not-relied-on. Nothing has been switched over yet; the checked-in
  default still routes everything to Fireworks. See DECISIONS.md D26 for the
  full quote and reasoning, and Next below for what actually unblocks now.
- **Step 0 (capture reality): done, 2026-07-09.**
  - Real `ALLOWED_MODELS` confirmed and **live-tested**, not just read:
    `gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it-nvfp4`,
    `minimax-m3`, `kimi-k2p7-code`. `minimax-m3` and `kimi-k2p7-code` both
    answered real questions correctly over the real Fireworks endpoint
    (D22). The 3 Gemma variants need an on-demand deployment first (per
    user) — deliberately not wired in, nothing depends on them.
  - **Found and fixed a real bug in the process**: Fireworks 404s on the
    bare model names `ALLOWED_MODELS` gives (`"Model not found,
    inaccessible, and/or not deployed"`) — it requires the full
    `accounts/fireworks/models/<name>` path (D23). Would have silently
    broken the actual submission. New `routing_eval/modelids.py`
    (`split_models` + `normalize_model_id`, extracted to avoid a
    conformance.py ↔ policy.py import cycle, same pattern as `taskio.py`)
    fixes this at both entry points — `ALLOWED_MODELS` parsing and
    hand-authored `routing_policy.json` "model" fields. Idempotent, so safe
    regardless of whether launch-day `ALLOWED_MODELS` ships bare or
    prefixed.
  - Real 8 practice tasks are in: `scripts/fixtures/practice_tasks.json`
    (renamed from `placeholder_tasks.json`). Schema confirmed
    `{task_id, prompt}` exactly — matches `taskio.py`'s existing
    `_PROMPT_FIELDS` first-choice guess, no code change needed there (D24).
    **The placeholder-fixture caveat is gone** — this is real data now, not
    a stand-in. Downstream consequence: the real tasks carry no
    category/gold/scorer, so `probe-local`/`bakeoff`/`generate-policy`
    against this file see everything as `category="uncategorized"` — that
    was always the honest state of the real contract, not a new gap.
- **Step 1 (conformance shell) dry run against real infra: done, 2026-07-09
  (D25).** Not submitted — that's still your call.
  - Real image built (`docker buildx build --platform linux/amd64 --load`):
    **43.4MB**, far under the 10GB cap.
  - Ran end-to-end via `scripts/conformance_smoke.sh` in its new
    `REAL_FIREWORKS=1` mode (added this session — same script now serves
    both the stub `routing:verify` gate and a real-Fireworks dry run) against
    all 8 real practice tasks, using the **checked-in safe-default policy
    unmodified** (`model: null` → resolved to `minimax-m3`, first of the
    scoped `ALLOWED_MODELS`). Exit 0, valid `results.json`, wall-clock
    **14s** (budget: 600s / 10 min, D19). Container output byte-identical
    to a local (non-container) run of the same policy.
  - **Total tokens across all 8 tasks: 1,370 at the time (superseded —
    see D27 above, now 1,537 after the prompt-template fix)** (per-task:
    150, 152, 147, 213, 176, 182, 165, 185) — raw baseline data, zero
    optimization applied, NOT a performance claim (D8: this run predates
    D21's resolution, so every task still escalated to Fireworks; nothing
    routed locally). `/tmp/real_practice_results.json` on this machine now
    holds the POST-fix answers (overwritten by the later run) — spot-checked
    and, for practice-08 specifically, re-verified correct by executing the
    actual code, not eyeballing it. We don't grade our own accuracy — the
    real judge is external (D18).
  - Added token-usage logging to `PolicyRouter` (stderr only, doesn't touch
    the `{task_id, answer}` output contract) so every future run — not just
    this one — reports real per-task token cost without a second live call.
- **Step 3 — routing policy: unchanged this session, but now unblocked
  (D26)** (see prior entry below); the local tier can now actually be used,
  not just architected.
- **77 tests pass (was 74 at the start of this session), zero runtime
  dependencies. `routing:verify`: PASS** — confirmed explicitly mid-session
  (74/74, before the bake-off code changes) and again at the end (77/77,
  after). Lint: same 2 pre-existing findings, no new ones. Container gate:
  exit 0. New this session: +3 modelselect tests for classifier-driven
  categorization in the bake-off (one of which reproduced D29's exact
  collision bug while being written, see above).
- Known, untouched: `ruff check` flags the same 2 pre-existing issues as
  every prior session (E741 in `gates/signals.py:81`, F401 in
  `llm/client.py:10`). Not blocking.
- **Still nothing committed** — fifth session in a row on an uncommitted
  tree. `git status` now also shows new `prompts.py`,
  `scripts/classifier_paraphrase_check.py`,
  `scripts/fixtures/classifier_paraphrases.json`. Recommend committing soon;
  the working tree is large enough that losing it would hurt.

### Prior session recap (Step 3, unchanged this session)

`routing_eval/policy.py` is wired into the `score` path: task → classify
(`classify.py`, cheap keyword classifier, pluggable) → policy lookup by
category → local-with-timeout-fallback or Fireworks call →
`{task_id, answer}`. `routing_policy.json` shape: `{category: {tier, model,
max_tokens, prompt_template, timeout_s?}}` + `"_default"`. Checked-in safe
default routes everything to Fireworks, `model: null`. `routing-eval
generate-policy` builds a draft from Step 2's `probe-local`/`bakeoff`
outputs. Local-call timeout-and-fallback proven by unit test. Full detail in
DECISIONS.md D17–D21 (architecture) and D26 (D21's resolution).

## Next (immediate)

- **[YOU] — highest priority, blocking submission**: eyeball the 8 answers
  in the table above (or `/tmp/real_practice_results.json` on this
  machine). If they look right, submit manually — nothing has been pushed
  to a registry or the platform. If you want the exact `docker push`
  commands, tell me the target GitHub owner/repo and I'll give you them.
- **[YOU / DISCORD]**: find out the real graded task set's category
  taxonomy (if any) before trusting the classifier on it. D29 showed
  `DEFAULT_KEYWORDS` is ~8% accurate on anything that isn't the literal 8
  practice prompts — if the real categories differ even slightly in
  phrasing, most tasks will silently land in `"uncategorized"` (safe —
  falls to `"_default"`/Fireworks, still correctly formatted now that
  `_default` is also pinned to kimi) or, worse, in the WRONG category (gets
  the wrong prompt template, e.g. a real code task not getting `code_only`
  because the classifier called it something else).
- **[YOU, when ready]** Supply the real 2–3B 4-bit GGUF local model — the
  other lever, now that D21 is resolved. Point `probe-local --local-base-url`
  at it once serving.
- **[ENG, now unblocked by D26, D30 gives it real numbers to compare
  against]** Once the local model is available: run `probe-local` for real
  (categories now classified, not "uncategorized" — but see D29's caveat on
  how much to trust them), then `routing-eval generate-policy` using this
  session's real bake-off records + local results. Review the draft before
  swapping it in. New baseline to beat: **763 tokens** (was 1,537 before
  D30/D32), so the local tier's marginal win is smaller now — still worth
  it since local is zero tokens regardless, but temper expectations of a
  dramatic further drop.
- **[ENG]** D29 is a real risk to the submission's category-dependent
  formatting (code_only, sentiment_with_justification), not just an
  accuracy nicety — a misclassified code task gets the generic template and
  reverts to the fenced/prose format D27 fixed. No fix designed yet
  (explicitly out of scope this session); options for next time: broaden
  keyword sets further, add a confidence-based fallback that's more
  conservative, or replace the classifier entirely (e.g. a cheap local-model
  classification call, still zero tokens per D17 if it never produces the
  final answer, D21a). Optional P3 polish: the 2 ruff findings.
- **[INTEL]** Token-count definition (prompt/completion/total) is the one
  Known Unknown still fully open — nothing new this session.
- **[YOU, low priority]** No GHCR/registry reference exists yet — this repo
  has no git remote. Push to GitHub whenever convenient; not blocking the
  dry-run work, only blocking an exact `docker push` command.

## Open questions / blockers

- **Highest priority now**: nothing technical — waiting on your go-ahead
  after eyeballing the 8 answers. See State above for the table.
- **Significant (D29, still unresolved)**: classifier generalization is ~8%
  on paraphrased tasks — a material risk to category-dependent prompt
  formatting on the real (unseen) graded task set, not just a nice-to-have
  accuracy stat. Unaffected by this session's changes either way.
- No GHCR/registry reference exists — no git remote configured on this repo
  yet. Low priority; doesn't block the dry-run work above.
- Token-count definition still open (CONTEXT.md Known Unknown #2).
- Local GGUF model (2–3B, 4-bit) not yet supplied — the other lever on an
  actually token-optimized submission (D21 resolved, D26).
- 3 Gemma `ALLOWED_MODELS` variants need an on-demand deployment before
  they're callable — deferred as a bonus decision, not blocking.
- `low_confidence_threshold` default (0.5) is an unvalidated placeholder —
  matters more now that the local tier is live; revisit once a broader real
  category taxonomy exists to check it against (current one is 8 examples,
  and D29 shows even that doesn't generalize).
- Live smoke (`scripts/live_smoke.py`) — separate from this session's dry
  run — still untouched; superseded in practice by the real conformance dry
  run above, but not formally re-verified via that specific script.

## Calendar

Today **2026-07-09** → deadline **Jul 11** (2 days left).

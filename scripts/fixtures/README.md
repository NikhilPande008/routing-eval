# Fixtures

`practice_tasks.json` is the **real** 8 practice tasks from the Track 1
participant guide, pasted in 2026-07-09 — `{task_id, prompt}` only, matching
the confirmed real `/input/tasks.json` schema exactly (`taskio.py`'s
`_PROMPT_FIELDS` already checked `"prompt"` first, so no code change was
needed). Used by `scripts/conformance_smoke.sh` (I/O shape, exit code
against the stub server) and as `DEFAULT_TASKS` for the `probe-local` /
`bakeoff` / `generate-policy` CLI commands.

This file was a fabricated placeholder (6 made-up items with invented
`category`/`gold`/`scorer` fields) until 2026-07-09, when it was replaced
with the real tasks and renamed from `placeholder_tasks.json`. The real
tasks carry no `category`, `gold`, or `scorer` — those were only ever our
own synthetic scaffolding for exercising `modelselect.py` before real data
existed, and the real contract doesn't provide them (accuracy is graded by
an external LLM-judge, D18, not our own `scorers.py`). Consequence:
`probe-local`/`bakeoff`/`generate-policy` run against this file now will
treat every task as `category="uncategorized"`, `gold=None` — connectivity
and plumbing are still provable, but any reported "accuracy" against this
file is meaningless until real categories (and some real judging signal) are
figured out. See DECISIONS.md D21/D21c for the open conflict over whether
local answers can be scored at all.

---
name: routing:end-session
description: Run the routing project's end-of-session routine — verify, update the handoff docs, commit, and print the next-session prompt. Use when the user says "end session", "I'm done", "wrap up", "commit and finish", "close this out", or "/routing:end-session", and offer it proactively if the user signals goodbye after a session that changed files. Ensures the baton is clean so the next session (on either surface) starts cold without re-deriving anything.
---

# routing:end-session

Leave the repo in a state where the next session — Claude Code or Cowork — starts
cold and correct. Do not skip steps; a dropped baton is the failure mode this whole
split exists to prevent.

## Steps

1. **Verify.** Run `routing:verify`. If it fails, stop — fix or explicitly record
   the known-failing state in HANDOFF.md before committing. Never commit a red
   build silently.

2. **Log decisions.** Any fork resolved this session goes into DECISIONS.md as a
   dated entry with a one-line why. Performance claims must cite a frontier number
   (records.json + report), per the trusted-artifact rule.

3. **Update HANDOFF.md:**
   - `Last updated` timestamp + who.
   - **State** — what changed this session (one or two lines).
   - **Next (immediate)** — the very next actions, tagged `[ENG]` / `[INTEL]` /
     `[YOU]`.
   - **Open questions / blockers** — refresh; clear anything resolved.
   - Update the calendar line if the date advanced relative to Jul 6 / Jul 11.

4. **Commit:**
   ```bash
   git add -A
   git commit -m "<concise summary of the session's change>"
   ```
   (Do not push, create PRs, change access, or delete history — those are outside
   this routine.)

5. **Print the next-session prompt** — a 2–4 line block the user can paste to open
   the next session with full context, e.g.:
   > Resume the routing agent. Read HANDOFF.md and CONTEXT.md first. Next up:
   > <the top item from HANDOFF "Next">. Surface: <Claude Code | Cowork>.

## Pass criteria

Verify green (or its failure explicitly logged), HANDOFF.md current, decisions
recorded, one clean commit, next-session prompt printed.

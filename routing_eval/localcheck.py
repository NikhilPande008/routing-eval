"""Deterministic validators for LOCAL-tier answers (2026-07-11, deadline-day
local tier). Local answers cost zero remote tokens (D26) but come from a small
bundled model, so before a local answer is accepted it must pass a cheap,
deterministic shape check; any violation escalates the task to Fireworks.

The asymmetry that makes this safe: a REJECTED local answer only costs tokens
(the task pays the same remote price it would have paid without a local tier);
an ACCEPTED bad answer costs accuracy. So every check below is tuned strict --
over-escalation is the cheap failure mode.

Pure stdlib, no model calls -- these run in-process on the grading VM.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from typing import Optional

# Refusals/AI-boilerplate a small model can emit on an odd prompt. Any of
# these in a local answer means "let the big model handle it".
_REFUSAL_RE = re.compile(
    r"\bi can(?:no|')t\b|\bi cannot\b|\bi'm sorry\b|\bi am sorry\b"
    r"|\bas an ai\b|\bi(?: a|')m unable\b|\bi am unable\b"
    r"|\bi don't have (?:access|enough)\b",
    re.IGNORECASE,
)

_SENTIMENT_LABELS = ("positive", "negative", "mixed", "neutral")
# Structural both-sides markers (rubric alignment): a Mixed justification
# must contain one of these to count as acknowledging both sides.
_CONTRAST_RE = re.compile(
    r"\bbut\b|\bwhile\b|\bhowever\b|\bthough\b|\byet\b|\bwhereas\b"
    r"|\bboth\b|\bon the other hand\b|\bdespite\b|\balthough\b",
    re.IGNORECASE)
# Praise vocabulary: inside a NEGATIVE-labeled justification this signals a
# mixed review hiding under a one-sided label (public-validation T03b:
# "Negative: ... despite praising the device's flawless functionality" is the
# rubric's explicit auto-fail). Deliberately NOT a both-sides marker for the
# Mixed check -- praise alone is one-sided there.
_PRAISE_RE = re.compile(r"\bprais|\bflawless\b|\bgreat\b|\bexcellent\b|\bfantastic\b"
                        r"|\blove[sd]?\b|\bamazing\b", re.IGNORECASE)

# entity line: optional bullet, entity text, separator (-, --, em/en dash, or
# colon), then an UPPERCASE type label. The 17/19 run's live answers used
# single-hyphen "Maria Sanchez-PERSON" style, so a single hyphen is accepted.
_ENTITY_LINE_RE = re.compile(
    r"^\s*(?:[-*•·]\s+)?(.+?)\s*(?:--+|[-–—:])\s*"
    r"([A-Z][A-Z_]{1,24}(?: [A-Z_]{2,24})?)\s*\.?\s*$"
)
# Official NER label set (Judging FAQ v2): one mislabel is tolerated live,
# two+ = fail. The local validator is STRICTER than the rubric on purpose --
# any unofficial label escalates (escalation costs tokens, never accuracy).
_OFFICIAL_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "DATE"}

# "exactly N words" / "in N words" / "no more than N words" / "under N words"
_EXACT_WORDS_RE = re.compile(r"\bexactly\s+(\d{1,3})\s+words\b", re.IGNORECASE)
_MAX_WORDS_RE = re.compile(
    r"\b(?:in|within|at most|no more than|under|fewer than|less than|maximum(?: of)?)"
    r"\s+(\d{1,3})\s+words\b", re.IGNORECASE)
_ONE_SENTENCE_RE = re.compile(r"\b(?:in |as )?(?:exactly )?(?:one|a single|1)\s+sentence\b",
                              re.IGNORECASE)
_N_SENTENCES_RE = re.compile(r"\b(?:in |within )?(?:\d\s*[-–]\s*)?(\d)\s+sentences\b",
                             re.IGNORECASE)
# Bullet constraints (rubric alignment, Judging FAQ v2: "exactly N bullets"
# and per-bullet word caps are HARD pass/fail). Word-number fallbacks cover
# "three bullets" etc.
_N_BULLETS_RE = re.compile(
    r"\b(?:exactly\s+)?(\d{1,2}|one|two|three|four|five|six)\s+bullet(?:s|\s+points?)?\b",
    re.IGNORECASE)
_BULLET_WORD_CAP_RE = re.compile(
    r"\b(?:at most|no more than|under|fewer than|maximum(?: of)?|up to)\s+(\d{1,3})\s+words?"
    r"\s+(?:per|each|in each|for each)\s+bullet"
    r"|\b(?:each|every|per)\s+bullet[^.?!]{0,40}?"
    r"\b(?:at most|no more than|under|fewer than|maximum(?: of)?|up to)\s+(\d{1,3})\s+words?\b",
    re.IGNORECASE)
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•·]|\d{1,2}[.)])\s+")
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _words(text: str) -> int:
    return len(text.split())


def _sentences(text: str) -> int:
    parts = [p for p in re.split(r"[.!?]+(?:\s+|$)", text.strip()) if p.strip()]
    return len(parts)


def _generic_problem(answer: str) -> Optional[str]:
    stripped = answer.strip()
    if not stripped:
        return "blank answer"
    if _REFUSAL_RE.search(stripped):
        return "refusal/AI-boilerplate"
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    for a, b, c in zip(lines, lines[1:], lines[2:]):
        if a == b == c:
            return "degenerate repetition loop"
    return None


def _sentiment_problem(prompt: str, answer: str) -> Optional[str]:
    stripped = answer.strip().lstrip("*#").strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    head = first_line[:80].lower()
    if not any(lbl in head for lbl in _SENTIMENT_LABELS):
        return "no sentiment label in the first line"
    if _words(stripped) < 3:
        return "label without justification"
    if len(stripped) > 800:
        return "answer too long for a label+clause task"
    # Rubric alignment (Judging FAQ v2): for a mixed review the justification
    # MUST acknowledge BOTH sides -- a one-sided reason fails regardless of
    # label. Heuristic: a Mixed-labeled answer without any contrast marker
    # reads one-sided; escalate (tokens, not accuracy).
    if "mixed" in head and not _CONTRAST_RE.search(stripped):
        return "Mixed label without a both-sides justification"
    # "Negative" on a mixed review is an explicit rubric FAIL, and the public
    # validation set caught the bundled 3B doing exactly that ("Negative: ...
    # despite praising the device's flawless functionality"). A Negative label
    # whose own justification contains a contrast/praise marker is a suspected
    # mixed review -- escalate rather than risk the auto-fail.
    if "negative" in head and (_CONTRAST_RE.search(stripped) or _PRAISE_RE.search(stripped)):
        return "Negative label with a both-sides justification (mixed review suspected)"
    # Input-side fence (D52, guards the 1.5B's known one-sided-justification
    # failure): if the TASK text itself carries a contrast marker (reads
    # mixed) and the label is Negative, escalate regardless of what the
    # justification says. Over-escalation on genuinely negative reviews that
    # happen to contain "but" costs tokens only.
    if "negative" in head and _CONTRAST_RE.search(prompt):
        return "Negative label on contrast-bearing (mixed-looking) input"
    # Rubric: on a mixed review the justification must acknowledge BOTH sides
    # REGARDLESS of label -- validation task T03 caught a one-sided
    # "Positive: the product is fantastic and support was helpful" on a
    # mixed review, an official rubric FAIL despite the accepted label. If
    # the input reads mixed, require a contrast marker in the answer too.
    if _CONTRAST_RE.search(prompt) and not _CONTRAST_RE.search(stripped):
        return "one-sided justification on contrast-bearing (mixed-looking) input"
    return None


def _entity_problem(prompt: str, answer: str) -> Optional[str]:
    lines = [ln for ln in answer.strip().splitlines() if ln.strip()]
    prompt_lower = prompt.lower()
    valid = 0
    other = 0
    entity_texts = []
    for ln in lines:
        m = _ENTITY_LINE_RE.match(ln.strip())
        if not m:
            # allow a single header line ("Here are the entities:" etc.)
            other += 1
            continue
        entity = m.group(1).strip().strip("\"'").rstrip(".,")
        if entity.lower() not in prompt_lower:
            return f"entity not found in the text: {entity!r}"
        if m.group(2) not in _OFFICIAL_ENTITY_TYPES:
            return f"unofficial entity label: {m.group(2)!r}"
        entity_texts.append((entity, m.group(2)))
        # Merged-entity fence (judge-proxy caught the 1.5B emitting
        # "United Nations Climate Summit in Geneva -- LOCATION": in-text and
        # officially labeled, but it swallows a separate entity -- missing
        # ANY entity is a rubric fail). Prepositional glue or >5 words in
        # one entity almost always means two entities fused.
        if re.search(r"\b(?:in|at|on|of the)\s", entity + " ") and len(entity.split()) > 3:
            return f"suspected merged entities: {entity!r}"
        if len(entity.split()) > 5:
            return f"entity text too long ({len(entity.split())} words): {entity!r}"
        valid += 1
    if valid == 0:
        return "no 'entity -- TYPE' lines"
    if other > 1:
        return f"{other} non-entity lines (prose instead of a list)"
    # Split-entity fence (2026-07-13, official T05: the 1.5B emitted
    # "March 15 -- DATE" and "2023 -- DATE" where "March 15 2023" is ONE
    # entity -- a rubric fail). Only fires on SAME-TYPE entities that are
    # space-contiguous in the text: two DATEs (or two of any type) that join
    # into a single span were almost certainly one entity wrongly split.
    # Different-type adjacency ("March 15 2023, Sundar Pichai") is normal and
    # must NOT trip this. Escalation is accuracy-safe (remote gets it right).
    for a, ta in entity_texts:
        for b, tb in entity_texts:
            if a is b or ta != tb:
                continue
            if f"{a} {b}".lower() in prompt_lower:
                return f"suspected split {ta} entity: {a!r} + {b!r} contiguous in the text"
    return None


def _summarization_problem(prompt: str, answer: str) -> Optional[str]:
    stripped = answer.strip()
    # Bullet constraints first (rubric: HARD pass/fail). When the task is
    # bullet-shaped, sentence/whole-answer word checks don't apply -- bullets
    # break sentence counting, and "N words per bullet" would false-trigger
    # the whole-answer _MAX_WORDS_RE.
    mb = _N_BULLETS_RE.search(prompt)
    if mb:
        want = _WORD_NUMBERS.get(mb.group(1).lower()) or int(mb.group(1))
        bullets = [ln for ln in stripped.splitlines() if _BULLET_LINE_RE.match(ln)]
        if len(bullets) != want:
            return f"{len(bullets)} bullets where exactly {want} were asked"
        mc = _BULLET_WORD_CAP_RE.search(prompt)
        if mc:
            cap = int(mc.group(1) or mc.group(2))
            for b in bullets:
                text = _BULLET_LINE_RE.sub("", b)
                if _words(text) > cap:
                    return f"bullet exceeds {cap}-word cap: {_words(text)} words"
        return _generic_problem(stripped)   # bullet checks replace prose checks
    m = _EXACT_WORDS_RE.search(prompt)
    if m and _words(stripped) != int(m.group(1)):
        return f"word count {_words(stripped)} != exactly {m.group(1)}"
    m = _MAX_WORDS_RE.search(prompt)
    if m and _words(stripped) > int(m.group(1)):
        return f"word count {_words(stripped)} > limit {m.group(1)}"
    if _ONE_SENTENCE_RE.search(prompt) and _sentences(stripped) > 1:
        return f"{_sentences(stripped)} sentences where one was asked"
    m = _N_SENTENCES_RE.search(prompt)
    if m and _sentences(stripped) > int(m.group(1)):
        return f"{_sentences(stripped)} sentences > limit {m.group(1)}"
    if _words(prompt) >= 60 and _words(stripped) >= _words(prompt):
        return "summary is not shorter than the source text"
    if _words(stripped) > 220:
        return "summary too long"
    return None


# ---------------------------------------------------------------------------
# Code (D53): a local code answer is kept ONLY with execution proof -- the
# task prompt must contain extractable input->output examples and the
# generated code must reproduce them. Tasks without parseable examples always
# escalate (kimi is the proven code answerer). Execution is the one validator
# STRONGER than the remote model: it can't be fooled by plausible-looking
# wrong code.
# ---------------------------------------------------------------------------

# "fn(args) should return X" / "returns X" / "-> X" / "== X" / "→ X"
_EXAMPLE_RE = re.compile(
    r"(\w+)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*"
    r"(?:should\s+(?:return|give|output|yield)|returns?|outputs?|->|→|==|=>)\s*"
    r"(\"[^\"]*\"|'[^']*'|\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|[-\w.]+)",
    re.IGNORECASE)

_HARNESS = r"""
import json, sys
spec = json.load(sys.stdin)
ns = {}
try:
    exec(spec["code"], ns)
except Exception as e:
    print(json.dumps({"error": f"exec failed: {e}"})); sys.exit(0)
results = []
for c in spec["checks"]:
    fn = ns.get(c["fn"])
    if not callable(fn):
        results.append(f"function {c['fn']} not defined"); continue
    try:
        got = fn(*c["args"])
    except Exception as e:
        results.append(f"{c['fn']} raised {type(e).__name__}: {e}"); continue
    if got != c["expected"] and str(got) != str(c["expected"]):
        results.append(f"{c['fn']}({c['args']}) -> {got!r}, expected {c['expected']!r}")
print(json.dumps({"failures": results}))
"""


def _extract_examples(prompt: str) -> list:
    checks = []
    for fn, args_s, expected_s in _EXAMPLE_RE.findall(prompt):
        try:
            args = list(ast.literal_eval(f"({args_s},)")) if args_s.strip() else []
            expected = ast.literal_eval(expected_s)
        except (ValueError, SyntaxError):
            continue   # unparseable example -> not usable as proof
        checks.append({"fn": fn, "args": args, "expected": expected})
    return checks


def _strip_fence(text: str) -> str:
    m = re.match(r"\s*```[\w]*\n(.*?)```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


def _code_problem(prompt: str, answer: str) -> Optional[str]:
    """Per-sample validation: shape + syntax always; example execution when
    the task carries parseable examples (hard evidence, free). Tasks WITHOUT
    examples pass through -- the two-sample DIFFERENTIAL execution check
    (agreement_problem below) is what verifies those."""
    code = _strip_fence(answer).strip()
    if not code:
        return "blank code"
    first = next((ln for ln in code.splitlines() if ln.strip()), "")
    if not re.match(r"\s*(def |class |import |from |@|#|[\w.\[\]]+\s*=)", first):
        return "prose contamination before code"
    try:
        compile(code, "<local-code>", "exec")
    except SyntaxError as e:
        return f"syntax error: {e.msg} (line {e.lineno})"
    checks = _extract_examples(prompt)
    if not checks:
        # D53 final: NO keeping without hard evidence. The differential-
        # execution variant leaked twice in gating (the 1.5B writes the SAME
        # canonical-but-wrong solution under both prompt wordings, and
        # identical wrong code agrees with itself) -- spec-style tasks
        # without parseable examples always escalate to kimi.
        return "no verifiable input/output examples in the task"
    try:
        proc = subprocess.run([sys.executable, "-c", _HARNESS],
                              input=json.dumps({"code": code, "checks": checks}),
                              capture_output=True, text=True, timeout=5)
        verdict = json.loads(proc.stdout.strip() or "{}")
    except (subprocess.TimeoutExpired, ValueError):
        return "execution check timed out or produced no verdict"
    if verdict.get("error"):
        return f"execution check: {verdict['error']}"
    if verdict.get("failures"):
        return f"execution check failed: {verdict['failures'][0]}"
    return None


# Differential execution (D53): run BOTH code samples on a synthesized input
# battery; keep only if every comparable call agrees. Batteries cover the
# task shapes the competition uses (list utilities, string predicates, small
# ints, two-string comparisons). A battery "applies" when both samples accept
# it without raising; >=3 agreeing calls with >=1 non-None output = proof.
_DIFF_HARNESS = r"""
import json, sys
spec = json.load(sys.stdin)
BATTERIES = {
    1: [
        [[1, 2, 3, 4, 5, 5]], [[5, 1, 4, 1, 5, 9, 2, 6]], [[3, 3, 3]],
        [[2, 1]], [[0, -1, -5, 10]], [[7]],
        ["A man a Plan"], ["racecar"], ["Hello World"], ["aa bb AB"],
        [0], [1], [5], [6],
    ],
    2: [
        ["listen", "silent"], ["Dormitory", "dirty room"], ["abc", "abd"],
        ["", ""], [[1, 2], [2, 1]], [3, 4],
    ],
}
def load(code):
    ns = {}
    exec(code, ns)
    fns = [v for k, v in ns.items() if callable(v) and not k.startswith("_")
           and getattr(v, "__module__", None) is None]
    return fns[-1] if fns else None
try:
    f1, f2 = load(spec["code1"]), load(spec["code2"])
except Exception as e:
    print(json.dumps({"error": f"exec failed: {e}"})); sys.exit(0)
if f1 is None or f2 is None:
    print(json.dumps({"error": "no function defined"})); sys.exit(0)
compared = agreed = nonnone = 0
disagreement = None
for args in BATTERIES.get(f1.__code__.co_argcount, []):
    if f1.__code__.co_argcount != len(args):
        continue
    try:
        r1 = f1(*[a.copy() if isinstance(a, list) else a for a in args])
        r2 = f2(*[a.copy() if isinstance(a, list) else a for a in args])
    except Exception:
        continue
    compared += 1
    if repr(r1) == repr(r2):
        agreed += 1
        if r1 is not None:
            nonnone += 1
    elif disagreement is None:
        disagreement = f"{args!r}: {r1!r} vs {r2!r}"
print(json.dumps({"compared": compared, "agreed": agreed,
                  "nonnone": nonnone, "disagreement": disagreement}))
"""


def code_agreement_problem(first: str, second: str) -> Optional[str]:
    c1, c2 = _strip_fence(first).strip(), _strip_fence(second).strip()
    try:
        proc = subprocess.run([sys.executable, "-c", _DIFF_HARNESS],
                              input=json.dumps({"code1": c1, "code2": c2}),
                              capture_output=True, text=True, timeout=8)
        v = json.loads(proc.stdout.strip() or "{}")
    except (subprocess.TimeoutExpired, ValueError):
        return "differential execution timed out or produced no verdict"
    if v.get("error"):
        return f"differential execution: {v['error']}"
    if v.get("disagreement"):
        return f"code samples disagree on {v['disagreement']}"
    if v.get("compared", 0) < 3 or v.get("nonnone", 0) < 1:
        return (f"insufficient differential evidence "
                f"({v.get('compared', 0)} comparable calls)")
    return None


_CATEGORY_CHECKS = {
    "sentiment": _sentiment_problem,
    "entity_extraction": _entity_problem,
    "summarization": _summarization_problem,
    "code": _code_problem,
    "code_debug": _code_problem,
    "code_gen": _code_problem,
}


def local_answer_problem(category: str, prompt: str, answer: str) -> Optional[str]:
    """None if the local answer is acceptable for this category; otherwise a
    short human-readable reason (which policy.py logs before escalating to
    Fireworks). Categories without a dedicated check still get the generic
    blank/refusal/loop screen."""
    problem = _generic_problem(answer)
    if problem is not None:
        return problem
    check = _CATEGORY_CHECKS.get(category)
    if check is not None:
        return check(prompt, answer)
    return None


# ---------------------------------------------------------------------------
# Two-sample self-consistency (D52): for categories whose validators cannot
# check CONTENT (knowledge: facts; math: arithmetic), a second local sample is
# drawn at a different temperature and the two must agree -- any disagreement
# escalates. The known danger case (both samples agreeing on the same wrong
# value) is accepted residual risk; disagreement catches the common one
# (observed live on validation task T02b: $4.50 vs $4.48).
# ---------------------------------------------------------------------------

# Tolerates "Answer:", "Final Answer:", "**Answer**:", "So the answer is:" --
# temp-0.7 samples vary the exact phrasing (observed on validation T02/T02b).
_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*[*#>\-\s]*(?:so\s+)?(?:the\s+)?(?:final\s+)?answer\b[^:=\n]*[:=]\s*(.+?)\s*$")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_STOPWORDS = frozenset(
    "the a an is are was were be been of to in on for and or it its this that "
    "with as by from at not no which who whom whose what when where how why "
    "does do did can could will would should has have had".split())


def _final_numbers(answer: str) -> Optional[list]:
    """The numbers on the final 'Answer:' line, normalized (commas stripped,
    trailing zeros canonicalized). None if no Answer line or no numbers."""
    matches = _ANSWER_LINE_RE.findall(answer)
    if not matches:
        return None
    nums = _NUM_RE.findall(matches[-1])
    if not nums:
        return None
    out = []
    for n in nums:
        n = n.replace(",", "")
        try:
            out.append(f"{float(n):g}")
        except ValueError:
            return None
    return out


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower())
            if w not in _STOPWORDS and len(w) > 2}


def agreement_problem(category: str, first: str, second: str) -> Optional[str]:
    """None if two local samples agree enough to trust; else the reason to
    escalate. math: the final Answer-line numbers must match EXACTLY (every
    requested value -- multi-part rubric). knowledge/general: content-word
    overlap (Jaccard) must clear a floor tuned to catch divergent facts, not
    paraphrase variance."""
    if category == "math":
        a, b = _final_numbers(first), _final_numbers(second)
        if a is None or b is None:
            return "no extractable Answer line in a math sample"
        if a != b:
            return f"math samples disagree: {a} vs {b}"
        return None
    if category in ("code", "code_debug", "code_gen"):
        return code_agreement_problem(first, second)
    if category == "logic":
        la = _ANSWER_LINE_RE.findall(first)
        lb = _ANSWER_LINE_RE.findall(second)
        if not la or not lb:
            return "no extractable Answer line in a logic sample"
        na = re.sub(r"[^\w\s]", "", la[-1].lower()).strip()
        nb = re.sub(r"[^\w\s]", "", lb[-1].lower()).strip()
        if na == nb or na in nb or nb in na:
            return None
        return f"logic samples disagree: {la[-1]!r} vs {lb[-1]!r}"
    wa, wb = _content_words(first), _content_words(second)
    if not wa or not wb:
        return "empty sample"
    jaccard = len(wa & wb) / len(wa | wb)
    # 0.2: calibrated so honest paraphrases of the same facts pass (measured
    # ~0.23 on same-content rewordings) while topic-level divergence is
    # caught (~0.03 on unrelated answers). Known limitation, accepted: this
    # catches an unstable/rambling model, NOT a single flipped fact that both
    # samples agree on -- that residual risk is what the live gate measures.
    if jaccard < 0.2:
        return f"samples diverge (overlap {jaccard:.2f})"
    return None

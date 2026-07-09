"""Category classifier for the routing policy (routing_eval.policy).

The I/O contract (D20) confirms task_id and nothing else about a task's
nature -- production tasks.json has no category field to read. The routing
policy needs a category to look up a policy entry, so something has to guess
one from the prompt text itself.

2026-07-09 (D37): the 8-way KeywordClassifier below is RETIRED from the
deployed path. D35 showed kimi-k2p7-code isn't weaker than minimax-m3 in any
category, so an 8-way split by category no longer buys anything on the model
side -- the only thing a category choice still controls is which of 3 prompt
templates (`code_only` / `sentiment_with_justification` / generic `default`)
gets used. D29/D36 showed the 8-way classifier does not generalize (8% hit
rate on paraphrases, 22% misroute even on a friendly hand-written set) --
replacing it with an 8-way anything was never going to fix that; the fix is
to stop needing 8 categories at all. TwoWayClassifier below detects only the
two things that change the answer's required FORMAT -- "is this code" and
"is this sentiment" -- each as an OR of a few independent, mostly-disjoint
signals (embedded code syntax; an action-verb + code-noun pair; an explicit
sentiment/tone/opinion phrase) rather than single-keyword hits, specifically
to avoid D29's generic-word-collision failure mode (a single word like
"what"/"how" hijacking classification). Everything else routes to the
generic template -- deliberately, since D35 found no category where the
generic template needs to be more specific than "code" or
"sentiment-with-justification" to get a comparable-or-better answer.
Verified against `scripts/fixtures/accuracy_diagnostic.json` (the 32-task
diagnostic) and `scripts/fixtures/classifier_paraphrases.json` (36
keyword-avoiding paraphrases) -- see `scripts/two_way_detector_check.py` and
DECISIONS.md D37 for the measured false-positive/false-negative rates.

KeywordClassifier/DEFAULT_KEYWORDS are KEPT, not deleted -- they're still
exercised by tests and by the bake-off/generate-policy tooling that predates
this change, and nothing here rules out a future category-specific need. They
are simply no longer PolicyRouter's default classifier (see policy.py).

KeywordClassifier is the cheapest thing that could work: match words against
a per-category keyword set and return the top match plus a margin-based
confidence (1.0 = the runner-up category matched nothing, 0.0 = tied with the
runner-up -- pure guesswork). Swap in something smarter later (a fine-tuned
classifier, a cheap local-model call) by implementing the same Classifier
protocol; nothing else in policy.py needs to change.

DEFAULT_KEYWORDS started as a fabricated starter set (math / knowledge /
wordplay) before any real data existed. Calibrated 2026-07-09 against the
real 8 practice tasks (scripts/fixtures/practice_tasks.json) in two passes:
`sentiment`/`code_debug`/`code_gen` first (so those 3 route to the right
`prompt_template` in `policy.py` -- code_only / sentiment_with_justification
-- after a real dry run showed the generic template producing wrong-format
answers), then `math`/`summarization`/`entity_extraction`/`logic` to cover
the remaining 5 practice tasks (`knowledge` already covered practice-01 and
was left as-is). Every keyword below was chosen because it's a literal
token in one specific practice task's prompt and absent from the other 7 --
verified by direct classification of all 8 real prompts, not assumed. This
is calibration against 8 known examples, not a general-purpose taxonomy;
expect false positives/negatives on tasks that don't resemble these 8.
`wordplay` has no matching practice task and is untested against real data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

_WORD = re.compile(r"[a-z0-9']+")

DEFAULT_KEYWORDS: Dict[str, List[str]] = {
    # practice-01 (capital of Australia)
    "knowledge": ["capital", "color", "colour", "name", "who", "what", "when",
                 "where", "country", "month", "year", "city"],
    # practice-02 (store item math word problem) -- "how" is the only literal
    # hit in that prompt; the original plus/minus/etc. keywords assume
    # spelled-out arithmetic words a word problem may not use.
    "math": ["plus", "minus", "times", "divided", "sum", "product", "equals",
            "calculate", "add", "subtract", "multiply", "total", "how"],
    "wordplay": ["spell", "backwards", "reverse", "rhyme", "anagram", "letter",
                "letters", "word"],
    # practice-03 (battery/screen review)
    "sentiment": ["sentiment"],
    # practice-04 (city council budget paragraph -> one sentence). Two
    # keywords needed: "summarize" alone ties 1-1 against knowledge's "city".
    "summarization": ["summarize", "summarise", "summary", "sentence"],
    # practice-05 (Maria Sanchez / Fireworks AI / Berlin)
    "entity_extraction": ["extract", "entities", "entity", "named"],
    # practice-06 (buggy get_max function)
    "code_debug": ["bug", "debug", "broken", "fix"],
    # practice-07 (Sam/Jo/Lee pet puzzle) -- "own"/"owns" both present as
    # distinct tokens, so this outscores knowledge's lone "who" hit.
    "logic": ["own", "owns", "different", "friends", "puzzle", "riddle"],
    # practice-08 (write a second-largest function) -- deliberately excludes
    # "function", which also appears in practice-06's code_debug prompt.
    "code_gen": ["write", "python", "implement"],
}


@dataclass
class ClassificationResult:
    category: str
    confidence: float                          # 0..1, margin-based (see KeywordClassifier)
    matched: Dict[str, int] = field(default_factory=dict)   # per-category hit counts, for logging


class Classifier(Protocol):
    def classify(self, prompt: str) -> ClassificationResult: ...


class KeywordClassifier:
    def __init__(self, keyword_map: Optional[Dict[str, List[str]]] = None):
        self.keyword_map = keyword_map or DEFAULT_KEYWORDS

    def classify(self, prompt: str) -> ClassificationResult:
        words = set(_WORD.findall(prompt.lower()))
        scores = {cat: len(words & set(kw)) for cat, kw in self.keyword_map.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_cat, top_score = ranked[0]
        if top_score == 0:
            return ClassificationResult("uncategorized", 0.0, scores)
        second_score = ranked[1][1] if len(ranked) > 1 else 0
        confidence = (top_score - second_score) / top_score
        return ClassificationResult(top_cat, confidence, scores)


# ---------------------------------------------------------------------------
# D37: two narrow, robust detectors -- the deployed replacement for the 8-way
# classifier above. Each is an OR of a few phrase/structure-level signals
# (never a single generic word) so a false positive requires more than one
# common word to collide, the specific failure mode D29 documented.
# ---------------------------------------------------------------------------

# Literal code pasted into the prompt (code_debug always does this: the bug
# has to be shown to be fixed). Natural-language prompts essentially never
# contain a real function signature, indexing, a method call, or a
# for-loop/list-comprehension by accident, so this signal alone is
# high-precision. Broadened beyond `def`/`return` (2026-07-09, D37 stress
# test) after finding those two alone miss embedded code that avoids a
# top-level function definition (e.g. a bare statement fragment).
_CODE_SYNTAX_RE = re.compile(
    r"```"                              # markdown fence
    r"|\bdef\s+\w+\s*\("                # python function signature
    r"|\breturns?\s+\w"                 # a `return`/`returns` statement
    r"|\bwhile\s+\w+\s*[<>=!]"          # a while-loop condition
    r"|:\s*\n\s+\S"                     # colon + newline + indent (code block shape)
    r"|\bfor\s+\w+\s+in\s+\w+"          # for-loop / list-comprehension clause
    r"|\w\[\w*\]"                       # bracket indexing, e.g. items[idx]
    r"|\.\w+\("                         # dot method call, e.g. nums.reverse(
    r"|=\s*\w+\s+if\b.{0,40}\belse\b",  # inline conditional assignment
    re.IGNORECASE,
)

# Nouns that are almost unambiguous on their own in a task-request context
# (unlike "code"/"program"/"method", which have common non-code senses).
_CODE_STRONG_NOUN_RE = re.compile(
    r"\bpython\b|\bfunction\b|\bscript\b|\balgorithm\b|\bexecutable\b", re.IGNORECASE)

# Weaker nouns that need an action verb alongside them to count -- avoids
# e.g. "zip code" or "loyalty program" alone triggering a false positive.
_CODE_WEAK_NOUN_RE = re.compile(r"\bcode\b|\bprogram\b|\bmethod\b", re.IGNORECASE)
_CODE_VERB_RE = re.compile(
    r"\bwrite\b|\bcreate\b|\bgenerate\b|\bproduce\b|\bimplement\b|\bbuild\b"
    r"|\bdesign\b|\bneed\b|\bmake\b|\bfix\b|\bdebug\b",
    re.IGNORECASE,
)


def is_code_task(prompt: str) -> bool:
    """True if the prompt pastes literal code, or names a code-domain noun
    (function/script/algorithm/python on its own, code/program/method only
    alongside a request verb like write/create/fix)."""
    if _CODE_SYNTAX_RE.search(prompt):
        return True
    if _CODE_STRONG_NOUN_RE.search(prompt):
        return True
    return bool(_CODE_VERB_RE.search(prompt) and _CODE_WEAK_NOUN_RE.search(prompt))


# Explicit sentiment/tone/opinion phrasing. Deliberately phrase-level (not
# bare "positive"/"negative", which collide with ordinary number/direction
# language elsewhere, e.g. math's "positive integer") to keep this narrow.
# The praise/complaint/upbeat/... word list was added after a held-out
# stress test (D37, scripts/fixtures/two_way_stress_test.json) found common
# "is this praise or a complaint" - style rephrasings missed the original
# five phrase patterns; each word added is itself fairly unambiguous
# feedback/opinion vocabulary, not a generic word (avoiding D29's exact
# failure mode), but see DECISIONS.md D37 for the honesty caveat on reusing
# the same fixture to both find and patch gaps.
_SENTIMENT_RE = re.compile(
    r"\bsentiment\b"
    r"|\btone\b"
    r"|\bopinion\b"
    r"|positive\s+or\s+negative|negative\s+or\s+positive"
    r"|favorable\s+or\s+unfavorable|unfavorable\s+or\s+favorable"
    r"|\bhow\s+(?:does|do)\b.{0,30}?\bfeels?\b"
    r"|\bfeel(?:s|ing)?\s+about\b"
    r"|\bpraise\b|\bcomplaint\b|\bcompliment\b|\bgripe\b"
    r"|\brave\b|\brant\b|\bupbeat\b|\bdownbeat\b"
    r"|\bunhappy\b|\bdisliked\b|\bupset\b",
    re.IGNORECASE,
)


def is_sentiment_task(prompt: str) -> bool:
    """True if the prompt explicitly asks for sentiment/tone/opinion, via a
    phrase-level match rather than a single generic word."""
    return bool(_SENTIMENT_RE.search(prompt))


class TwoWayClassifier:
    """The deployed classifier (D37): only detects the two things that
    change which prompt template gets used -- code and sentiment. Everything
    else is "general", which resolves through policy.py's "_default" entry
    (the generic template) exactly like an unmatched category always has.
    Confidence is always 1.0 -- these are deterministic yes/no detectors,
    not a margin-based guess, so there's no partial-confidence state to
    report; "general" is a deliberate default, not an uncertain one."""

    def classify(self, prompt: str) -> ClassificationResult:
        if is_code_task(prompt):
            return ClassificationResult("code", 1.0, {"code": 1})
        if is_sentiment_task(prompt):
            return ClassificationResult("sentiment", 1.0, {"sentiment": 1})
        return ClassificationResult("general", 1.0, {})

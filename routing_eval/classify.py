"""Category classifier for the routing policy (routing_eval.policy).

The I/O contract (D20) confirms task_id and nothing else about a task's
nature -- production tasks.json has no category field to read. The routing
policy needs a category to look up a policy entry, so something has to guess
one from the prompt text itself.

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

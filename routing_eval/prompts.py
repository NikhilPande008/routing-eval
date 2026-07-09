"""Named system prompts, shared by policy.py (the routing policy, Step 3)
and modelselect.py (the bake-off, Step 2) -- both need to know which system
prompt a category maps to. Lives in its own leaf module (no internal
dependencies beyond llm.runners) because policy.py already depends on
modelselect.py (for LocalViability/ModelCategoryRanking), so modelselect.py
importing policy.py for this would cycle -- same problem, same fix pattern
as taskio.py and modelids.py.

code_only / sentiment_with_justification added 2026-07-09 after the real
conformance dry run: DEFAULT_SYSTEM's "briefly, do not explain" is fine for
a bare fact but actively wrong for code (the model wraps in a markdown
fence -- not runnable as-is) and sentiment (it drops the justification the
guide asks for). Format instructions, not a model-competence issue -- see
DECISIONS.md D27.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

from .llm.runners import DEFAULT_SYSTEM

CODE_ONLY_SYSTEM = (
    "Output ONLY runnable code that directly answers the request -- no "
    "markdown code fences (no ``` anywhere), no prose before or after, no "
    "explanation of the approach. A short comment inside the code is fine. "
    "The entire response must be valid, directly executable code."
)
SENTIMENT_SYSTEM = (
    "Classify the sentiment. Respond in exactly this form: the label "
    "(Positive, Negative, Mixed, or Neutral), then a colon, then a "
    "one-sentence justification grounded in specific details from the text. "
    "Do not add anything else."
)

PROMPT_TEMPLATES: Dict[str, Optional[str]] = {
    "default": DEFAULT_SYSTEM,
    "code_only": CODE_ONLY_SYSTEM,
    "sentiment_with_justification": SENTIMENT_SYSTEM,
}


def system_prompt(template_name: str) -> Optional[str]:
    return PROMPT_TEMPLATES.get(template_name, DEFAULT_SYSTEM)


def load_category_templates(policy_path: str) -> Dict[str, str]:
    """category -> prompt_template name, read directly as raw JSON from a
    routing_policy.json file. Deliberately doesn't go through policy.py's
    PolicyEntry/load_policy (that would need policy.py, which depends on
    modelselect.py -- the same cycle this module exists to avoid). Only the
    "prompt_template" field is used; tier/model/timeout are policy.py's
    concern, irrelevant to picking a system prompt for a bake-off call."""
    with open(policy_path) as f:
        raw = json.load(f)
    return {cat: entry.get("prompt_template", "default") for cat, entry in raw.items()}

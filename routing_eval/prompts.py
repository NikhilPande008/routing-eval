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

2026-07-10 (D40): "default" no longer maps to DEFAULT_SYSTEM (llm/runners.py's
terse "answer only, as briefly as possible, do not explain"). Submission #2
scored 63.2% (12/19, up 1 task from the classifier fix) -- terseness is the
next-strongest suspect for the remaining misses: the judge grades intent
(D18), and DEFAULT_SYSTEM was written for token minimization, actively wrong
for a category like "explain this concept" where a correct answer requires
2-5 sentences, not a bare fact. ACCURATE_GENERIC_SYSTEM replaces it as the
"default" template -- ONE prompt, not a return to per-category classification
(that's exactly the 8-way-classifier mistake D37 just retired): it
instructs the MODEL to pick the right response shape (explanation length,
reasoning trace, stated format constraint) from the task's own content,
since the model can read task intent far better than a keyword classifier
ever could. CODE_ONLY_SYSTEM / SENTIMENT_SYSTEM are UNCHANGED (explicit
instruction -- those two are not implicated in the terseness failure mode,
they already require multi-part output).
"""
from __future__ import annotations

import json
from typing import Dict, Optional

from .llm.runners import DEFAULT_SYSTEM  # noqa: F401 -- kept as the raw runner-level fallback

# 2026-07-10 (D42, after the D41 trim FAILED live at 73.7% / 14/19 despite
# a 32/32 + 8/8 judge-proxy sweep -- the proxy is leniently miscalibrated
# and is now directional-only, never sufficient to ship a trim): this is the
# ONE conservative intermediate step between the 17/19 config (D40, 6,351
# live tokens) and the failed full trim. The D41 minimal math/entity/logic
# templates are RETIRED (their text and measured savings live in DECISIONS.md
# D41); math/logic go back to the verbatim 17/19-era full template
# (ACCURATE_FULL_SYSTEM below), entity gets an explicit-type-label template,
# and only the default takes a halfway trim (~235 -> ~120 tokens,
# instruction-shortening only -- every content requirement kept).
CODE_ONLY_SYSTEM = (
    "Output only runnable code -- no markdown code fences, no comments, "
    "no prose before or after."
)
SENTIMENT_SYSTEM = (
    "Reply with exactly: the sentiment label (Positive, Negative, Mixed, or "
    "Neutral), a colon, then one short clause of justification from the text."
)
# The D40 template, byte-identical to the one inside the 17/19 image
# (extracted from `nikhilpande/routing-eval:submission-17of19`, not
# rewritten from memory). math and logic route here: the 17/19 run is the
# only live-validated treatment those categories have, so they get it
# verbatim -- including the answer-first calculation wording whose
# intermittent logic flake (D40) the 17/19 run survived. D41's answer-last
# logic fix was only ever proxy-validated and shipped inside the run that
# failed live, so it does not override a live-validated config.
ACCURATE_FULL_SYSTEM = (
    "Answer accurately and completely. Correctness matters far more than "
    "brevity -- do not sacrifice information the question needs just to be "
    "short. Choose your answer's shape based on what's actually being "
    "asked:\n"
    "- If the question asks you to explain a concept, fact, or how "
    "something works, give a complete, direct explanation in 2-5 sentences "
    "-- a single word or fragment is not acceptable when explanation was "
    "requested.\n"
    "- If the question is a calculation or word problem, state the final "
    "answer clearly, then add a one-line reasoning trace showing how you "
    "got it.\n"
    "- If asked to summarize or rewrite text under a stated length or "
    'format constraint (e.g. "in one sentence", "in exactly N words"), '
    "follow that constraint exactly as written -- do not exceed or fall "
    "short of it.\n"
    '- If the question has multiple parts (e.g. "what is X, and what is '
    'it near"), answer every part explicitly -- do not drop any part.\n'
    "- Otherwise, give the most direct correct answer, with brief "
    "supporting detail only where it helps confirm the answer is right.\n"
    "Do not add unrelated commentary or filler, but never omit information "
    "needed to fully and correctly answer what was asked."
)
# Entity extraction, restored from D41's 14-token one-liner toward the
# fuller treatment, with the type labels made explicit (the 17/19 run's
# entity answers all carried PERSON/ORGANIZATION/LOCATION/DATE-style labels).
ENTITY_TYPED_SYSTEM = (
    "List every named entity in the text, one per line, as: entity -- TYPE. "
    "Use explicit type labels such as PERSON, ORGANIZATION, LOCATION, DATE, "
    "TIME, MONEY, PERCENT, PRODUCT, EVENT. Include every entity mentioned; "
    "add nothing that is not in the text."
)
# 2026-07-11 (D47, PLAN-TOKEN-OPT.md Step D47): the default trimmed
# ~120 -> ~75 tokens. Every D40-critical behavioral requirement survives,
# and the exact phrases test_default_template_is_accurate_not_terse asserts
# ("2-5 sentences", "every part", "follow it exactly", banned terseness
# phrases absent) are preserved verbatim. This is the last worthwhile
# remote-side template trim -- the full lever analysis lives in the plan.
ACCURATE_GENERIC_SYSTEM = (
    "Answer accurately and completely -- correctness beats brevity.\n"
    "- Explaining a concept or fact: a direct explanation in 2-5 sentences, "
    "never a bare word.\n"
    "- A calculation: the final answer, then one line of working.\n"
    "- A stated length or format constraint: follow it exactly.\n"
    "- A multi-part question: answer every part.\n"
    "- Otherwise: the most direct correct answer.\n"
    "Do not add filler, but never omit information the question needs."
)

# 2026-07-10 (D43, authorized deviation from D42's conservatism, reasoning
# logged in DECISIONS.md): the Sam/Jo/Lee logic puzzle hit its wrong-
# headline-then-self-correct pattern 3/3 times under ACCURATE_FULL_SYSTEM's
# answer-first wording (near-deterministic, not the intermittent flake it
# was in the D40 era) -- and this exact deduction-first/answer-last wording
# already has 6/6 live A/B validation from D41, including 3/3 repeats of the
# specific flaky prompt, byte-identical output each repeat. Its only
# negative association is having shipped inside the D41 trim that failed
# live at 73.7% -- that failure is attributed to the OTHER D41 trims (math/
# entity/default), not this template, which was never itself implicated.
# Scope is deliberately narrow: ONLY "logic" repoints to this template;
# "math" stays on ACCURATE_FULL_SYSTEM (the verbatim, live-validated 17/19
# treatment) -- this text is not reused for math.
LOGIC_ANSWER_LAST_SYSTEM = (
    "Show your deduction in a few short steps, then end with 'Answer:' "
    "followed by the final answer."
)

# 2026-07-11 (local tier): system prompts for LOCAL calls. Local tokens are
# free (D26), so these are written for a small model's answer quality --
# explicit format rules plus a worked example, verbosity is costless here.
# Remote treatments are untouched: these are only referenced by policy
# entries' local_prompt_template, never by a Fireworks call.
SUMMARIZE_SYSTEM = (
    "Summarize the given text accurately. Follow any stated length or format "
    'constraint exactly as written ("in one sentence" means exactly one '
    'sentence; "in exactly N words" means count the words; "in N bullets" '
    "means exactly N lines each starting with '- ', respecting any per-bullet "
    "word limit). Cover both the positive and the negative points the text "
    "makes. Output only the summary itself -- no preamble, no commentary."
)
SENTIMENT_LOCAL_SYSTEM = (
    "You label the sentiment of a text. Reply with exactly one line: the "
    "label (Positive, Negative, Mixed, or Neutral), a colon, then one short "
    "clause of justification grounded in the text. Use Mixed when the text "
    "contains both praise and criticism.\n"
    "Example reply:\n"
    "Mixed: praises the camera quality but criticizes the short battery life."
)
# 2026-07-11 (rubric alignment, official Judging FAQ v2): the official NER
# label set is ONLY PERSON/ORGANIZATION/LOCATION/DATE -- one mislabel is
# tolerated, two+ = fail, missing any entity = fail. The earlier nine-type
# list invited unofficial labels (the 3B used EVENT on a fixture); this
# version constrains generation to the official four.
ENTITY_LOCAL_SYSTEM = (
    "Extract every PERSON, ORGANIZATION, LOCATION, and DATE entity from the "
    "user's text. Output one entity per line, exactly in the form: "
    "entity -- TYPE, using ONLY those four type labels (a company, agency, "
    "university, or institution is ORGANIZATION; a city, country, or "
    "geographic place is LOCATION; any date or time expression is DATE). "
    "Copy each entity "
    "exactly as it appears in the text. Include every such entity mentioned; "
    "add nothing that is not in the text; output no other lines.\n"
    "Example lines:\n"
    "Maria Sanchez -- PERSON\n"
    "Berlin -- LOCATION\n"
    "last March -- DATE"
)

# 2026-07-11 (D45, Step 2a): a SINGLE-category trim, deliberately isolated --
# after D45 showed a bundled multi-category change can't be localized
# post-hoc when it fails live (the exact D42 lesson, now reinforced), this
# session trims ONLY math and gates/ships it alone before touching anything
# else. math is the single most expensive live category under
# ACCURATE_FULL_SYSTEM (~361 tokens/task, D44's judge-proxy table) --
# highest payoff of any remaining single-category trim. Keeps the one shape
# requirement math answers have actually been judged PASS on across every
# prior live-validated run ("final answer, then one line of working" --
# e.g. practice-02's real D44 answer: "Final answer: 144 items remain. /
# Reasoning: ..."), drops the shared multi-part/summarization/explanation
# bullets that ACCURATE_FULL_SYSTEM carries for OTHER categories math never
# needed. "math" is the ONLY policy entry repointed to this template --
# "logic" stays on LOGIC_ANSWER_LAST_SYSTEM, everything else is untouched
# from D43.
MATH_DIRECT_SYSTEM = (
    "Solve the problem accurately -- correctness matters more than brevity. "
    "State the final answer clearly, then add one line showing the working "
    "that produced it. Do not skip a step needed to verify the answer."
)

# 2026-07-12 (D52 all-in redesign, user directive: top-3 needs <=2,024
# tokens, "if not, there's no point freezing to anything"): knowledge and
# math go LOCAL with two-sample self-consistency (localcheck.agreement_
# problem) fencing content. Local tokens are free -- these prompts are
# written for the 1.5B's answer quality, verbosity costless.
KNOWLEDGE_LOCAL_SYSTEM = (
    "Answer accurately and completely. Answer EVERY part of the question -- "
    "if it asks two things, answer both. For explain-questions give the "
    "mechanism or reason in 2-5 sentences, never just the bare fact. Do not "
    "add filler or disclaimers."
)
MATH_LOCAL_SYSTEM = (
    "Solve the math problem step by step, briefly. If the question asks for "
    "more than one quantity, compute each one. Your LAST line must be "
    "exactly this form, listing every requested value, and nothing may come "
    "after it:\n"
    "Answer: <value(s)>"
)
# D53: consistency via PROMPT diversity at temperature 0, not sampling noise.
# d52's live gates showed the temp-0.7 second sample injecting arithmetic
# errors and once answering only a sub-part -- a differently-worded prompt at
# temperature 0 decorrelates mistakes while correct answers still agree.
# DIVERSITY_PROMPT_TEMPLATES maps a category's local template to the
# alternate wording used for the SECOND self-consistency sample.
MATH_LOCAL_V2_SYSTEM = (
    "Work the problem again carefully. Write each intermediate result on its "
    "own line and double-check every multiplication and subtraction. Answer "
    "every part the question asks. Finish with one final line in exactly "
    "this form and write nothing after it:\n"
    "Answer: <value(s)>"
)
LOGIC_LOCAL_SYSTEM = (
    "Solve the puzzle by explicit deduction: list each clue's consequence in "
    "a short numbered step, then end with 'Answer:' followed by the final "
    "answer. Nothing may come after the Answer line."
)
LOGIC_LOCAL_V2_SYSTEM = (
    "Re-derive the solution from scratch. For each person or item, state "
    "what the clues force. Eliminate impossibilities one by one, then finish "
    "with one final line: 'Answer:' followed by the final answer, and write "
    "nothing after it."
)
CODE_LOCAL_SYSTEM = (
    "Write the requested Python code. Output ONLY runnable code -- no "
    "markdown fences, no comments, no prose before or after. Define exactly "
    "the function the task asks for."
)
CODE_LOCAL_V2_SYSTEM = (
    "Implement the requested Python function a careful, different way than "
    "the obvious first attempt. Handle edge cases: empty inputs, duplicates, "
    "mixed case, spaces, zero. Output ONLY runnable code -- no markdown "
    "fences, no comments, no prose."
)

PROMPT_TEMPLATES: Dict[str, Optional[str]] = {
    "default": ACCURATE_GENERIC_SYSTEM,
    "code_only": CODE_ONLY_SYSTEM,
    "sentiment_with_justification": SENTIMENT_SYSTEM,
    "accurate_full": ACCURATE_FULL_SYSTEM,
    "entity_typed": ENTITY_TYPED_SYSTEM,
    "logic_answer_last": LOGIC_ANSWER_LAST_SYSTEM,
    "math_direct": MATH_DIRECT_SYSTEM,
    "summarize_exact": SUMMARIZE_SYSTEM,
    "sentiment_local": SENTIMENT_LOCAL_SYSTEM,
    "entity_local": ENTITY_LOCAL_SYSTEM,
    "knowledge_local": KNOWLEDGE_LOCAL_SYSTEM,
    "math_local": MATH_LOCAL_SYSTEM,
    "logic_local": LOGIC_LOCAL_SYSTEM,
    "code_local": CODE_LOCAL_SYSTEM,
}

# The SECOND self-consistency sample's system prompt, keyed by the entry's
# local_prompt_template (D53). Missing key => policy.py falls back to the
# same prompt at temperature 0.7 (the D52 behavior).
DIVERSITY_PROMPT_TEMPLATES: Dict[str, str] = {
    "math_local": MATH_LOCAL_V2_SYSTEM,
    "logic_local": LOGIC_LOCAL_V2_SYSTEM,
    "code_local": CODE_LOCAL_V2_SYSTEM,
}


def system_prompt(template_name: str) -> Optional[str]:
    return PROMPT_TEMPLATES.get(template_name, ACCURATE_GENERIC_SYSTEM)


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

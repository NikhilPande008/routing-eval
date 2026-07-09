import json

from routing_eval.classify import (DEFAULT_KEYWORDS, KeywordClassifier, TwoWayClassifier,
                                   is_code_task, is_sentiment_task)
from routing_eval.taskio import load_tasks, task_prompt

PRACTICE_TASKS_PATH = "scripts/fixtures/practice_tasks.json"
ACCURACY_DIAGNOSTIC_PATH = "scripts/fixtures/accuracy_diagnostic.json"
PARAPHRASES_PATH = "scripts/fixtures/classifier_paraphrases.json"


def test_unambiguous_prompt_gets_high_confidence():
    clf = KeywordClassifier()
    r = clf.classify("Calculate 2 plus 2.")
    assert r.category == "math"
    assert r.confidence == 1.0    # "calculate"/"plus" hit math only, no runner-up


def test_no_keyword_hits_is_uncategorized_zero_confidence():
    clf = KeywordClassifier()
    r = clf.classify("Describe the weather today.")
    assert r.category == "uncategorized"
    assert r.confidence == 0.0


def test_tied_categories_yield_zero_confidence():
    clf = KeywordClassifier(keyword_map={"a": ["foo"], "b": ["foo"]})
    r = clf.classify("foo bar")
    assert r.confidence == 0.0     # tied with the runner-up -- pure guesswork
    assert r.category in ("a", "b")


def test_custom_keyword_map_is_pluggable():
    clf = KeywordClassifier(keyword_map={"custom": ["banana"]})
    r = clf.classify("I like banana bread")
    assert r.category == "custom"
    assert r.confidence == 1.0


def test_default_keywords_cover_all_8_calibrated_categories():
    assert set(DEFAULT_KEYWORDS) == {
        "math", "knowledge", "wordplay", "sentiment", "summarization",
        "entity_extraction", "code_debug", "logic", "code_gen",
    }


def test_all_8_real_practice_tasks_classify_into_their_intended_category():
    """Step 2 calibration (2026-07-09): every category here was chosen
    because its keyword is a literal token in exactly one of the 8 real
    practice tasks. code_debug/code_gen/sentiment additionally drive
    routing_policy.default.json's code_only / sentiment_with_justification
    templates -- a misclassification there sends the wrong system prompt."""
    expected = {
        "practice-01": "knowledge",
        "practice-02": "math",
        "practice-03": "sentiment",
        "practice-04": "summarization",
        "practice-05": "entity_extraction",
        "practice-06": "code_debug",
        "practice-07": "logic",
        "practice-08": "code_gen",
    }
    tasks = load_tasks(PRACTICE_TASKS_PATH)
    clf = KeywordClassifier()
    for i, t in enumerate(tasks):
        result = clf.classify(task_prompt(t, i))
        assert result.category == expected[t["task_id"]], (
            f"{t['task_id']} classified {result.category!r}, "
            f"expected {expected[t['task_id']]!r} (matched={result.matched})")


# ---------------------------------------------------------------------------
# D37: the two narrow detectors that replaced the 8-way classifier in the
# deployed path (PolicyRouter's default is now TwoWayClassifier, not
# KeywordClassifier). See DECISIONS.md D37 and scripts/two_way_detector_check.py
# for the full false-positive/false-negative measurement across
# accuracy_diagnostic.json + classifier_paraphrases.json.
# ---------------------------------------------------------------------------

def _load_fixture(path):
    with open(path) as f:
        return json.load(f)


def test_is_code_task_true_on_embedded_code_and_code_request_language():
    assert is_code_task("This function should return the max of a list but has a bug: "
                        "def get_max(nums): return nums[0]. Find and fix it.")
    assert is_code_task("Write a Python function that returns the second-largest number "
                        "in a list, handling duplicates correctly.")
    assert is_code_task("Create a function that checks whether a given string is a palindrome.")


def test_is_code_task_false_on_unrelated_prompts():
    assert not is_code_task("What is the capital of Australia, and what body of water is it near?")
    assert not is_code_task("A store has 240 items. It sells 15% on Monday. How many remain?")
    assert not is_code_task("A loyalty program gives customers 2 points per dollar. How many "
                           "points does a $30 purchase earn?")   # "program" alone, no verb


def test_is_sentiment_task_true_on_explicit_sentiment_phrasing():
    assert is_sentiment_task("Classify the sentiment of this review: great battery, bad screen.")
    assert is_sentiment_task("Is this restaurant review positive or negative: the food was great "
                            "but the service was slow?")
    assert is_sentiment_task("Determine the tone of this message: absolutely loved it!")


def test_is_sentiment_task_false_on_unrelated_prompts_including_math_with_positive():
    assert not is_sentiment_task("A shirt costs $40 and is discounted by 25%. What is a positive "
                                "integer solution for the remaining stock?")
    assert not is_sentiment_task("Extract all named entities and their types from: Maria Sanchez "
                                "joined Fireworks AI in Berlin last March.")


def test_two_way_classifier_returns_code_sentiment_or_general():
    clf = TwoWayClassifier()
    assert clf.classify("Write a Python function that reverses a string.").category == "code"
    assert clf.classify("Classify the sentiment: great phone, terrible battery.").category == "sentiment"
    assert clf.classify("What is the capital of France?").category == "general"


def test_two_way_classifier_zero_false_positives_or_negatives_on_accuracy_diagnostic():
    """32-task diagnostic (D35/D36): every code_debug/code_gen task must
    detect as 'code', every sentiment task must detect as 'sentiment', and
    none of the other 24 tasks should be misdetected as either."""
    tasks = _load_fixture(ACCURACY_DIAGNOSTIC_PATH)
    for t in tasks:
        category = TwoWayClassifier().classify(t["prompt"]).category
        if t["category"] in ("code_debug", "code_gen"):
            assert category == "code", f"{t['task_id']} ({t['category']}) detected {category!r}"
        elif t["category"] == "sentiment":
            assert category == "sentiment", f"{t['task_id']} detected {category!r}"
        else:
            assert category == "general", (
                f"{t['task_id']} ({t['category']}) unexpectedly detected as {category!r}")


def test_two_way_classifier_zero_false_positives_or_negatives_on_paraphrases():
    """D29's 36 keyword-avoiding paraphrases -- the set the 8-way classifier
    scored only 8% on. The two-way detectors must still get every code_debug/
    code_gen/sentiment paraphrase right, with no false positives elsewhere."""
    cases = _load_fixture(PARAPHRASES_PATH)
    for c in cases:
        category = TwoWayClassifier().classify(c["prompt"]).category
        if c["category"] in ("code_debug", "code_gen"):
            assert category == "code", f"{c['prompt']!r} detected {category!r}"
        elif c["category"] == "sentiment":
            assert category == "sentiment", f"{c['prompt']!r} detected {category!r}"
        else:
            assert category == "general", f"{c['prompt']!r} unexpectedly detected as {category!r}"


def test_two_way_classifier_leaves_8_practice_tasks_routing_unchanged():
    """The 3 real practice tasks that need a non-generic template
    (code_debug/code_gen/sentiment) must still detect correctly under the
    new classifier, same as they did under the retired KeywordClassifier."""
    expected = {
        "practice-03": "sentiment", "practice-06": "code", "practice-08": "code",
    }
    tasks = load_tasks(PRACTICE_TASKS_PATH)
    clf = TwoWayClassifier()
    for i, t in enumerate(tasks):
        if t["task_id"] not in expected:
            continue
        result = clf.classify(task_prompt(t, i))
        assert result.category == expected[t["task_id"]]

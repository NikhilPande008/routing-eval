from routing_eval.classify import DEFAULT_KEYWORDS, KeywordClassifier
from routing_eval.taskio import load_tasks, task_prompt

PRACTICE_TASKS_PATH = "scripts/fixtures/practice_tasks.json"


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

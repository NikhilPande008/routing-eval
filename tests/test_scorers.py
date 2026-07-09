from routing_eval import scorers
from routing_eval.schema import Item


def s(pred, item):
    return scorers.score(pred, item)


def test_numeric_extracts_last_number():
    it = Item("1", "math", "q", 42, "numeric", {"rel_tol": 1e-6})
    assert s("The answer is 42.", it) == 1.0
    assert s("first 3, then 5, so 42", it) == 1.0          # takes the last number
    assert s("41", it) == 0.0
    assert s("no number here", it) == 0.0


def test_numeric_handles_commas_and_currency():
    it = Item("2", "math", "q", 1234, "numeric", {})
    assert s("$1,234", it) == 1.0


def test_exact_list_gold_and_normalization():
    it = Item("1", "qa", "q", ["yes", "yeah"], "exact", {})
    assert s("  Yeah ", it) == 1.0                     # case + whitespace normalized
    assert s("no", it) == 0.0
    assert s("Yeah!", it) == 0.0                        # punctuation NOT stripped by default
    it2 = Item("2", "qa", "q", ["yeah"], "exact", {"strip_punct": True})
    assert s("Yeah!", it2) == 1.0                       # ...but it is when asked


def test_multiple_choice_takes_last_label():
    it = Item("1", "cls", "q", "positive", "multiple_choice", {},
              allowed=["positive", "negative", "neutral"])
    assert s("negative at first, but overall positive", it) == 1.0
    assert s("negative", it) == 0.0


def test_token_f1_partial_credit():
    it = Item("1", "qa", "q", "Grace", "token_f1", {})
    assert s("Grace", it) == 1.0
    assert s("Frank", it) == 0.0
    it2 = Item("2", "qa", "q", "the annual budget report", "token_f1", {})
    v = s("annual report", it2)
    assert 0.0 < v < 1.0


def test_json_match_exact_and_subset():
    it = Item("1", "extract", "in", {"a": 1, "b": 2}, "json_match", {"mode": "exact"})
    assert s('{"a": 1, "b": 2}', it) == 1.0
    assert s('```json\n{"a": 1, "b": 2}\n```', it) == 1.0
    assert s('{"a": 1}', it) == 0.0
    it2 = Item("2", "extract", "in", {"a": 1}, "json_match", {"mode": "subset"})
    assert s('{"a": 1, "b": 9}', it2) == 1.0

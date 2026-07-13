"""localcheck.py: deterministic shape checks that gate LOCAL-tier answers.
Rejection escalates to Fireworks (costs tokens); acceptance of a bad answer
is the only accuracy-costing failure mode, so checks are strict."""
from routing_eval.localcheck import local_answer_problem

SENTIMENT_PROMPT = ("What is the sentiment of this review? 'The battery life is "
                    "amazing but the screen scratches easily.'")
ENTITY_PROMPT = ("Extract all named entities from: Maria Sanchez joined "
                 "Fireworks AI in Berlin last March.")
SUMMARY_PROMPT = ("Summarize the following in one sentence: The city council "
                  "voted on Tuesday to approve the new budget, which increases "
                  "school funding by 12 percent while cutting road maintenance.")


# -- generic screens (any category) -----------------------------------------

def test_blank_answer_rejected():
    assert local_answer_problem("sentiment", SENTIMENT_PROMPT, "   ") is not None


def test_refusal_rejected():
    assert local_answer_problem("summarization", SUMMARY_PROMPT,
                                "I'm sorry, as an AI I cannot summarize this.") is not None


def test_repetition_loop_rejected():
    loopy = "The budget passed.\n" * 5
    assert local_answer_problem("summarization", SUMMARY_PROMPT, loopy) is not None


def test_unknown_category_gets_generic_screen_only():
    assert local_answer_problem("knowledge", "capital of France?", "Paris.") is None
    assert local_answer_problem("knowledge", "capital of France?", "") is not None


# -- sentiment ---------------------------------------------------------------

def test_sentiment_label_with_justification_accepted():
    assert local_answer_problem(
        "sentiment", SENTIMENT_PROMPT,
        "Mixed: praises the battery life but criticizes the screen.") is None


def test_sentiment_markdown_and_prose_label_accepted():
    assert local_answer_problem(
        "sentiment", SENTIMENT_PROMPT,
        "**Mixed** -- the reviewer praises battery life but dislikes the screen.") is None


def test_sentiment_missing_label_rejected():
    assert local_answer_problem(
        "sentiment", SENTIMENT_PROMPT,
        "The reviewer seems to have complicated feelings about the product.") is not None


def test_sentiment_bare_label_without_justification_rejected():
    assert local_answer_problem("sentiment", SENTIMENT_PROMPT, "Mixed") is not None


# -- entity extraction -------------------------------------------------------

def test_entity_typed_lines_accepted():
    answer = ("Maria Sanchez -- PERSON\nFireworks AI -- ORGANIZATION\n"
              "Berlin -- LOCATION\nlast March -- DATE")
    assert local_answer_problem("entity_extraction", ENTITY_PROMPT, answer) is None


def test_entity_single_hyphen_and_bullets_accepted():
    answer = ("- Maria Sanchez-PERSON\n- Fireworks AI-ORGANIZATION\n"
              "- Berlin-LOCATION\n- last March-DATE")
    assert local_answer_problem("entity_extraction", ENTITY_PROMPT, answer) is None


def test_entity_hallucinated_entity_rejected():
    answer = "Maria Sanchez -- PERSON\nGoogle -- ORGANIZATION"
    problem = local_answer_problem("entity_extraction", ENTITY_PROMPT, answer)
    assert problem is not None and "Google" in problem


def test_entity_prose_instead_of_list_rejected():
    answer = ("The text mentions a person named Maria Sanchez who joined a "
              "company called Fireworks AI in the city of Berlin.")
    assert local_answer_problem("entity_extraction", ENTITY_PROMPT, answer) is not None


# -- summarization -----------------------------------------------------------

def test_one_sentence_summary_accepted():
    assert local_answer_problem(
        "summarization", SUMMARY_PROMPT,
        "The city council approved a budget raising school funding by 12 percent "
        "while cutting road maintenance.") is None


def test_multi_sentence_summary_rejected_when_one_asked():
    assert local_answer_problem(
        "summarization", SUMMARY_PROMPT,
        "The council voted on Tuesday. The budget raises school funding. "
        "Road maintenance was cut.") is not None


def test_exact_word_count_enforced():
    prompt = "Summarize this in exactly 5 words: The cat sat on the mat all day long."
    assert local_answer_problem("summarization", prompt, "Cat sat on mat daily.") is None
    assert local_answer_problem("summarization", prompt,
                                "The cat sat on the mat.") is not None


def test_max_word_count_enforced():
    prompt = "Summarize in no more than 10 words: " + "The merger fell through. " * 10
    assert local_answer_problem("summarization", prompt, "The merger fell through.") is None
    twelve = "word " * 12
    assert local_answer_problem("summarization", prompt, twelve) is not None


# -- rubric alignment (official Judging FAQ v2) -------------------------------

def test_entity_unofficial_label_rejected():
    prompt = "Extract entities: The summit in Geneva was hosted by MIT on March 15."
    ok = "Geneva -- LOCATION\nMIT -- ORGANIZATION\nMarch 15 -- DATE"
    assert local_answer_problem("entity_extraction", prompt, ok) is None
    bad = "Geneva -- LOCATION\nMIT -- ORGANIZATION\nthe summit -- EVENT"
    problem = local_answer_problem("entity_extraction", prompt, bad)
    assert problem is not None and "EVENT" in problem


def test_bullet_count_enforced():
    prompt = ("Summarize the report in exactly 3 bullets: Revenue grew fast "
              "but costs also rose sharply across all divisions this year.")
    good = "- Revenue grew fast\n- Costs rose sharply\n- All divisions affected"
    assert local_answer_problem("summarization", prompt, good) is None
    two = "- Revenue grew fast\n- Costs rose sharply"
    assert local_answer_problem("summarization", prompt, two) is not None


def test_bullet_word_cap_enforced():
    prompt = ("Summarize in exactly 2 bullets, no more than 5 words per bullet: "
              "The launch succeeded but reviews were harsh about the price.")
    good = "- Launch succeeded\n- Reviews criticized price"
    assert local_answer_problem("summarization", prompt, good) is None
    long_bullet = ("- The launch succeeded beyond all expectations this quarter\n"
                   "- Reviews harsh")
    assert local_answer_problem("summarization", prompt, long_bullet) is not None


def test_mixed_label_requires_both_sides_justification():
    prompt = "Sentiment of: 'Great camera but the battery dies fast.'"
    both = "Mixed: praises the camera but criticizes the battery life."
    assert local_answer_problem("sentiment", prompt, both) is None
    one_sided = "Mixed: the camera quality is praised."
    assert local_answer_problem("sentiment", prompt, one_sided) is not None


def test_negative_label_with_contrast_justification_rejected():
    """Public validation T03b caught the 3B labeling a mixed tweet Negative
    while its own justification acknowledged the praise -- the exact rubric
    auto-fail ('Negative on a mixed review = FAIL')."""
    prompt = "Classify: 'Box came dented but the device works flawlessly.'"
    bad = ("Negative: The tweet mentions dented box and missing manual, "
           "despite praising the device's flawless functionality.")
    assert local_answer_problem("sentiment", prompt, bad) is not None
    # D52 input-side fence: on a contrast-bearing (mixed-looking) PROMPT, a
    # Negative label escalates even with a one-sided justification -- the
    # 1.5B's one-sided-reason failure mode is invisible to answer-side checks.
    one_sided = "Negative: the reviewer reports the box arrived dented."
    assert local_answer_problem("sentiment", prompt, one_sided) is not None
    # a Negative on a genuinely one-sided negative prompt stays local
    plain = "Classify: 'Terrible product. It broke on day one and support ignored me.'"
    ok = "Negative: the reviewer reports the product broke and support ignored them."
    assert local_answer_problem("sentiment", plain, ok) is None


# -- D52: two-sample self-consistency ----------------------------------------

def test_math_agreement_exact_numbers():
    from routing_eval.localcheck import agreement_problem
    a = "Step 1: 2400*0.63=1512. Step 2: +800=2312. Step 3: -640.\nAnswer: 1,672 units"
    b = "Working: 2400-888+800-640.\nAnswer: 1672"
    assert agreement_problem("math", a, b) is None          # 1,672 == 1672
    c = "Answer: 1670"
    assert agreement_problem("math", a, c) is not None      # disagree -> escalate
    d = "I think the answer is about 1672."                 # no Answer line
    assert agreement_problem("math", a, d) is not None


def test_math_agreement_multipart_all_values():
    from routing_eval.localcheck import agreement_problem
    a = "Scale by 1.5.\nAnswer: 1.875 cups, $4.50"
    b = "12/8*1.25 = 1.875; 1.875*2.40 = 4.50.\nAnswer: 1.875 cups; $4.50"
    assert agreement_problem("math", a, b) is None
    # the observed live failure: one sample computed $4.48
    c = "Answer: 1.875 cups, $4.48"
    assert agreement_problem("math", a, c) is not None


def test_knowledge_agreement_content_overlap():
    from routing_eval.localcheck import agreement_problem
    a = ("RAM is volatile working memory for programs in use; ROM is "
         "non-volatile permanent storage holding firmware like the BIOS.")
    b = ("RAM stores data for running programs and loses it without power, "
         "while ROM permanently stores firmware such as the BIOS.")
    assert agreement_problem("knowledge", a, b) is None
    divergent = ("The capital of Australia is Canberra, chosen as a compromise "
                 "between Sydney and Melbourne in 1908.")
    assert agreement_problem("knowledge", a, divergent) is not None


# -- D53: code execution proof + logic agreement ------------------------------

CODE_PROMPT = ("Write a function second_largest(numbers) that returns the second "
               "largest distinct value. For example second_largest([1, 2, 3, 4, 5, 5]) "
               "should return 4.")


def test_code_kept_only_with_passing_execution_proof():
    good = ("def second_largest(numbers):\n"
            "    uniq = sorted(set(numbers))\n"
            "    return uniq[-2] if len(uniq) >= 2 else None")
    assert local_answer_problem("code", CODE_PROMPT, good) is None
    wrong = ("def second_largest(numbers):\n"
             "    return sorted(numbers)[-2]")   # returns 5 on the example
    problem = local_answer_problem("code", CODE_PROMPT, wrong)
    assert problem is not None and "execution check failed" in problem


def test_code_without_examples_escalates():
    """D53 FINAL: no keeping without hard evidence. The differential-execution
    variant leaked twice in gating (same canonical-but-wrong solution under
    both prompt wordings agrees with itself) -- example-less code tasks
    always escalate to kimi. code_agreement_problem is kept (with its
    regression tests below) only for a possible future stronger local model."""
    prompt = "Write a function that parses a config file into a dict."
    code = "def parse(path):\n    return {}"
    problem = local_answer_problem("code", prompt, code)
    assert problem is not None and "no verifiable" in problem


def test_code_syntax_error_and_prose_rejected():
    assert "syntax error" in local_answer_problem("code", CODE_PROMPT,
                                                  "def broken(:\n    pass")
    assert "prose" in local_answer_problem("code", CODE_PROMPT,
                                           "Sure! Here is the function you asked for.")


def test_logic_agreement_on_answer_line():
    from routing_eval.localcheck import agreement_problem
    a = "1. Jo owns the dog.\n2. Sam not bird.\nAnswer: Sam owns the cat."
    b = "Eliminating: Jo=dog, Lee=bird.\nAnswer: Sam owns the cat"
    assert agreement_problem("logic", a, b) is None
    c = "Answer: Lee owns the cat."
    assert agreement_problem("logic", a, c) is not None
    d = "I believe it is Sam."   # no Answer line
    assert agreement_problem("logic", a, d) is not None


# -- D53: differential execution for example-less code tasks ------------------

def test_differential_agreement_correct_pair_kept():
    from routing_eval.localcheck import code_agreement_problem
    a = ("def second_largest(nums):\n"
         "    u = sorted(set(nums))\n"
         "    return u[-2] if len(u) > 1 else None")
    b = ("def second_largest(nums):\n"
         "    best = second = None\n"
         "    for n in nums:\n"
         "        if best is None or n > best:\n"
         "            second, best = best, n\n"
         "        elif n != best and (second is None or n > second):\n"
         "            second = n\n"
         "    return second")
    assert code_agreement_problem(a, b) is None


def test_differential_disagreement_escalates():
    from routing_eval.localcheck import code_agreement_problem
    right = ("def second_largest(nums):\n"
             "    u = sorted(set(nums))\n"
             "    return u[-2] if len(u) > 1 else None")
    wrong = ("def second_largest(nums):\n"
             "    return sorted(nums)[-2]")   # duplicates break it
    problem = code_agreement_problem(right, wrong)
    assert problem is not None and "disagree" in problem


def test_differential_helper_retained_for_future_use():
    """code_agreement_problem is no longer wired into the deployed path (code
    is example-proof-only after the same-wrong-twice leaks) but stays tested
    so a future stronger local model can re-enable it safely."""
    from routing_eval.localcheck import code_agreement_problem
    same = "def f(x):\n    return x * 2"
    assert code_agreement_problem(same, same) is None

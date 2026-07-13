import json

from routing_eval.llm import (LocalRunner, OpenAICompatibleClient, RemoteRunner,
                              StubClient, stub_response)
from routing_eval.schema import Item

ITEM = Item("1", "math", "What is 2 plus 2?", 4, "numeric", {})


def test_local_runner_parses_answer_logprobs_samples():
    client = StubClient(stub_response(
        contents=["4", "4", "5"],
        token_logprobs=[[-0.1], [-0.2], [-2.0]],
        usage={"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}))
    out = LocalRunner(client, "local", n_samples=3, temperature=0.7).run(ITEM)
    assert out.answer == "4"
    assert out.samples == ["4", "4", "5"]
    assert out.token_logprobs == [-0.1]
    assert out.tokens == 1
    # request carried logprobs + n
    assert client.calls[0]["logprobs"] is True
    assert client.calls[0]["n"] == 3


def test_remote_runner_reports_usage_and_minimal_call():
    client = StubClient(stub_response(
        contents=["4"], usage={"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}))
    out = RemoteRunner(client, "remote", max_tokens=8, stop=["\n"]).run(ITEM)
    assert out.answer == "4"
    assert (out.prompt_tokens, out.completion_tokens, out.total_tokens) == (20, 2, 22)
    call = client.calls[0]
    assert call["max_tokens"] == 8 and call["stop"] == ["\n"] and call["temperature"] == 0.0
    assert call["extra"] == {"reasoning_effort": "none"}      # billed reasoning suppressed


def test_remote_and_local_runner_default_max_tokens_is_512_not_32():
    """D40: the 32-token default was a token-minimization leftover that
    silently truncates multi-sentence answers -- a guaranteed judge failure
    (D18). 512 is a generous cap; truncation is the risk now, not tokens."""
    assert RemoteRunner(StubClient(stub_response(["x"])), "m").max_tokens == 512
    assert LocalRunner(StubClient(stub_response(["x"])), "m").max_tokens == 512


def test_remote_runner_reports_finish_reason():
    client = StubClient(stub_response(contents=["a truncated ans"], finish_reason="length"))
    out = RemoteRunner(client, "remote").run(ITEM)
    assert out.finish_reason == "length"

    client_ok = StubClient(stub_response(contents=["4"]))
    out_ok = RemoteRunner(client_ok, "remote").run(ITEM)
    assert out_ok.finish_reason == "stop"


class _FakeResp:
    def __init__(self, body):
        self._b = body.encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_real_client_builds_request_and_parses(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(json.dumps({"choices": [{"message": {"content": "4"}}],
                                     "usage": {"prompt_tokens": 5, "completion_tokens": 1,
                                               "total_tokens": 6}}))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    c = OpenAICompatibleClient("https://api.fireworks.ai/inference/v1", api_key="sk-test")
    resp = c.chat(model="m", messages=[{"role": "user", "content": "2+2"}],
                  max_tokens=8, stop=["\n"])
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "m" and captured["body"]["max_tokens"] == 8
    assert resp["choices"][0]["message"]["content"] == "4"

import pytest
from unittest.mock import patch, MagicMock
import json
from llmwrapper import LLMWrapper, Prompt


def make_response(content: str) -> bytes:
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": content}}]
    }).encode()


@pytest.fixture
def llm():
    return LLMWrapper(api_key="test-key")


def test_prompt_messages_no_system():
    p = Prompt(user="hello")
    assert p.to_messages() == [{"role": "user", "content": "hello"}]


def test_prompt_messages_with_system():
    p = Prompt(user="hello", system="be terse")
    msgs = p.to_messages()
    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_prompt_messages_with_history():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]
    p = Prompt(user="follow up", history=history)
    msgs = p.to_messages()
    assert msgs[0] == {"role": "user", "content": "hi"}
    assert msgs[-1] == {"role": "user", "content": "follow up"}


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_APIKEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_APIKEY"):
        LLMWrapper(api_key=None)


def test_chat_returns_content(llm):
    mock_resp = MagicMock()
    mock_resp.read.return_value = make_response("Hello!")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    prompt = "say hello"
    print(f"\n[PROMPT] {prompt}")
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = llm.chat(prompt)
    print(f"[RESPONSE] {result}")

    assert result == "Hello!"


def test_call_with_system(llm):
    mock_resp = MagicMock()
    mock_resp.read.return_value = make_response("Arrr!")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    p = Prompt(user="greet me", system="you are a pirate")
    print(f"\n[PROMPT] {p.to_messages()}")
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
        result = llm.call(p)
    print(f"[RESPONSE] {result}")

    assert result == "Arrr!"
    call_args = mock_url.call_args[0][0]
    body = json.loads(call_args.data)
    assert body["messages"][0] == {"role": "system", "content": "you are a pirate"}


def test_http_error_raises(llm):
    import urllib.error
    err = urllib.error.HTTPError(url="", code=401, msg="Unauthorized", hdrs={}, fp=None)
    err.read = lambda: b'{"error": "invalid key"}'

    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="401"):
            llm.chat("hello")


@pytest.mark.live
def test_live_chat():
    llm = LLMWrapper()
    prompt = "Reply with exactly one word: Hello"
    print(f"\n[PROMPT] {prompt}")
    result = llm.chat(prompt)
    print(f"[RESPONSE] {result}")
    assert isinstance(result, str) and len(result) > 0

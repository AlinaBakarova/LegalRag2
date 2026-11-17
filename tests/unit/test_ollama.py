import json

import httpx
import pytest
from pydantic import ValidationError

from legal_workspace.config import Settings
from legal_workspace.llm.ollama import LLMError, OllamaGenerator


@pytest.mark.parametrize(
    "status,code",
    [(404, "llm_model_missing"), (503, "llm_busy"), (429, "llm_busy"), (400, "llm_api_error")],
)
def test_http_errors_are_safe(status, code):
    generator = OllamaGenerator(
        Settings(_env_file=None),
        transport=httpx.MockTransport(lambda req: httpx.Response(status, text="PRIVATE CONTENT")),
    )
    with pytest.raises(LLMError, match=code) as error:
        generator.complete([], {}, max_tokens=10)
    assert "PRIVATE" not in str(error.value)


def test_timeout_and_bad_json():
    def timeout(req):
        raise httpx.ReadTimeout("PRIVATE", request=req)

    generator = OllamaGenerator(Settings(_env_file=None), transport=httpx.MockTransport(timeout))
    with pytest.raises(LLMError, match="llm_timeout"):
        generator.complete([], {}, max_tokens=10)
    generator.transport = httpx.MockTransport(lambda req: httpx.Response(200, text="broken"))
    with pytest.raises(LLMError, match="llm_invalid_response"):
        generator.complete([], {}, max_tokens=10)


def test_local_model_is_configurable_and_thinking_disabled():
    def respond(req):
        assert req.url.host == "127.0.0.1"
        body = json.loads(req.content)
        assert body["model"] == "qwen3:8b"
        assert body["think"] is False and body["stream"] is False
        assert body["options"]["num_predict"] == 400
        return httpx.Response(
            200, json={"done": True, "message": {"content": "{}"}, "eval_count": 12}
        )

    generator = OllamaGenerator(
        Settings(_env_file=None, llm_model="qwen3:8b"), transport=httpx.MockTransport(respond)
    )
    assert generator.complete([], {}, max_tokens=400).output_tokens == 12


@pytest.mark.parametrize(
    "values",
    [
        {"ollama_url": "https://api.example.org"},
        {"ollama_url": "http://user:password@localhost:11434"},
        {"llm_model": "qwen3:cloud"},
        {"llm_model": "gpt-oss:120b-cloud"},
    ],
)
def test_remote_or_cloud_generation_is_rejected(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)

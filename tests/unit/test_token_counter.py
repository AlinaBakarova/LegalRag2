import json

import httpx
from tokenizers.pre_tokenizers import ByteLevel

from legal_workspace.config import Settings
from legal_workspace.llm.ollama import OllamaGenerator
from legal_workspace.llm.tokens import TokenCounter, qwen_counter


def metadata():
    # Synthetic byte-level vocabulary exercises the same BPE construction, not model quality.
    return {
        "model_info": {
            "tokenizer.ggml.pre": "qwen2",
            "tokenizer.ggml.tokens": sorted(ByteLevel.alphabet()),
            "tokenizer.ggml.merges": [],
        }
    }


def test_local_ollama_metadata_builds_counter_without_generation_or_downloads():
    requests = []

    def respond(request):
        requests.append(request)
        assert request.url.path == "/api/show"
        assert json.loads(request.content) == {"model": "qwen3:4b", "verbose": True}
        return httpx.Response(200, json=metadata())

    generator = OllamaGenerator(Settings(_env_file=None), transport=httpx.MockTransport(respond))
    counter = generator.token_counter()
    assert counter.method == "ollama_qwen_bpe_with_template_reserve"
    assert counter.count("Пример 123") == len("Пример 123".encode())
    assert len(requests) == 1


def test_tokenizer_failure_has_explicit_conservative_fallback():
    generator = OllamaGenerator(
        Settings(_env_file=None), transport=httpx.MockTransport(lambda req: httpx.Response(404))
    )
    counter = generator.token_counter()
    assert counter.method == "utf8_upper_bound"
    assert counter.count("Я") == 2


def test_qwen_merges_reduce_estimate_and_schema_counts_towards_budget():
    data = metadata()
    data["model_info"]["tokenizer.ggml.tokens"].append("ab")
    data["model_info"]["tokenizer.ggml.merges"] = ["a b"]
    counter = qwen_counter(data)
    assert counter.count("abab") == 2
    assert TokenCounter().count("abab") == 4
    messages = [{"role": "user", "content": "abab"}]
    assert counter.prompt(messages, {"type": "object"}) > counter.prompt(messages)

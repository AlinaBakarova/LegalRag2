import json
from dataclasses import dataclass
from typing import Protocol

import httpx


class LLMError(ValueError):
    """Safe public code; optional completion is for explicitly requested local diagnostics."""

    def __init__(self, code, *, completion=None):
        super().__init__(code)
        self.completion = completion


@dataclass(frozen=True)
class Completion:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ns: int = 0
    finish_reason: str | None = None


class Generator(Protocol):
    def complete(
        self,
        messages: list[dict],
        schema: dict,
        *,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> Completion: ...


class OllamaGenerator:
    def __init__(self, settings, *, transport=None):
        self.settings = settings
        self.transport = transport

    def complete(self, messages, schema, *, max_tokens, timeout_seconds=None):
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "keep_alive": "5m",
            "options": {
                "temperature": 0,
                "num_ctx": self.settings.llm_context_tokens,
                "num_predict": max_tokens,
            },
        }
        try:
            with httpx.Client(
                base_url=self.settings.ollama_url,
                timeout=httpx.Timeout(
                    min(
                        timeout_seconds or self.settings.llm_timeout_seconds,
                        self.settings.llm_timeout_seconds,
                    ),
                    connect=5,
                ),
                transport=self.transport,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                # Bound response bytes even if a server ignores stream=False/num_predict.
                with client.stream("POST", "/api/chat", json=payload) as response:
                    if response.status_code == 404:
                        raise LLMError("llm_model_missing")
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise LLMError("llm_busy")
                    if response.status_code != 200:
                        raise LLMError("llm_api_error")
                    body = bytearray()
                    for block in response.iter_bytes(8192):
                        body.extend(block)
                        if len(body) > 200_000:
                            raise LLMError("llm_invalid_response")
            result = json.loads(body)
            message = result["message"]
            content = message["content"]
            completion = Completion(
                content if isinstance(content, str) else "",
                int(result["prompt_eval_count"]) if "prompt_eval_count" in result else None,
                int(result["eval_count"]) if "eval_count" in result else None,
                int(result.get("total_duration", 0)),
                result.get("done_reason"),
            )
            if not result.get("done") or result.get("done_reason") == "length":
                raise LLMError("llm_output_limit", completion=completion)
            if not isinstance(content, str) or not content.strip():
                raise LLMError("llm_invalid_response", completion=completion)
            return completion
        except httpx.TimeoutException as exc:
            raise LLMError("llm_timeout") from exc
        except httpx.ConnectError as exc:
            raise LLMError("llm_unavailable") from exc
        except httpx.HTTPError as exc:
            raise LLMError("llm_api_error") from exc
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, LLMError):
                raise
            raise LLMError("llm_invalid_response") from exc

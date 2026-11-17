"""Mock LLM for reproducible RAG tests, never a substitute for the Ollama test."""

import json

from legal_workspace.llm.ollama import Completion


class MockLLM:
    def __init__(self, *, hook=None, invalid=False, refuse=False, review=False):
        self.calls = []
        self.hook = hook
        self.invalid = invalid
        self.refuse = refuse
        self.reject_review = review

    def complete(self, messages, schema, *, max_tokens, timeout_seconds=None):
        self.calls.append((messages, schema))
        if self.hook:
            self.hook(len(self.calls))
        if schema["title"] == "SupportReview":
            result = {
                "supported": not self.reject_review,
                "unsupported_claims": [0] if self.reject_review else [],
            }
        elif self.refuse:
            result = {"sufficient": False, "claims": [], "limitations": []}
        else:
            sources = json.loads(messages[1]["content"])["sources"]
            source = next((s for s in sources if "17 дней" in s["text"]), sources[0])
            quote = source["text"].strip()[:300]
            if "Срок оплаты составляет 17 дней." in source["text"]:
                quote = "Срок оплаты составляет 17 дней."
            result = {
                "sufficient": True,
                "claims": [
                    {
                        "text": quote,
                        "citations": [
                            {
                                "source_id": "S999" if self.invalid else source["source_id"],
                                "quote": quote,
                            }
                        ],
                    }
                ],
                "limitations": [],
            }
        return Completion(json.dumps(result, ensure_ascii=False), 100, 50, 1000000)

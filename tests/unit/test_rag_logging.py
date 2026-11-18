import json
import logging
import stat

import httpx
import pytest

from legal_workspace.config import Settings
from legal_workspace.llm.ollama import LLMError, OllamaGenerator
from legal_workspace.rag.pipeline import CallBudget, build_context
from legal_workspace.rag.trace import RagTrace, logger, rag_log_file


def events(caplog):
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == logger.name]


def source(chunk_id, text):
    return {"id": chunk_id, "title": "Тест", "heading": "", "text": text, "start": 0}


def test_context_accepts_more_than_four_sources_without_excerpts(caplog):
    caplog.set_level(logging.INFO, logger=logger.name)
    sources = [source(i, "Синтетический текст без юридических утверждений.") for i in range(1, 9)]
    _, contexts = build_context(
        Settings(_env_file=None), "Вопрос", [], sources, trace=RagTrace("m", "j")
    )
    event = events(caplog)[-1]
    assert event["candidate_chunk_ids"] == list(range(1, 9))
    assert event["selected_chunk_ids"] == list(range(1, 9))
    assert event["source_map"] == {f"S{i}": i for i in range(1, 9)}
    assert event["dropped"] == []
    assert event["trimmed"] == []
    assert "Синтетический текст" not in caplog.text
    assert [s["id"] for s in contexts.values()] == list(range(1, 9))


def test_context_skips_oversized_regions_without_arbitrary_trimming(caplog):
    caplog.set_level(logging.INFO, logger=logger.name)
    sources = [
        source(1, "Синтетический тестовый текст. " * 1000),
        source(2, "Завершённый короткий абзац с тестовым условием."),
    ]
    messages, contexts = build_context(
        Settings(_env_file=None), "Вопрос", [], sources, trace=RagTrace("m", "j")
    )
    event = events(caplog)[-1]
    assert event["trimmed"] == []
    assert event["dropped"] == [{"chunk_id": 1, "reason": "context_budget"}]
    assert event["selected_chunk_ids"] == [2]
    assert contexts["S1"]["text"] == sources[1]["text"]
    assert event["prompt_bytes"] == sum(len(m["content"].encode()) for m in messages)
    assert len(sources[0]["text"]) > 20000


def test_context_logs_all_discarded_before_raising(caplog):
    caplog.set_level(logging.INFO, logger=logger.name)
    with pytest.raises(LLMError, match="llm_context_limit"):
        build_context(
            Settings(_env_file=None), "Вопрос", [], [source(9, "short")], trace=RagTrace("m", "j")
        )
    assert events(caplog)[-1]["dropped"] == [{"chunk_id": 9, "reason": "too_short"}]
    assert events(caplog)[-1]["selected_chunk_ids"] == []


def test_ollama_output_limit_logs_partial_response_and_actual_counts(caplog):
    caplog.set_level(logging.INFO, logger=logger.name)
    settings = Settings(_env_file=None)
    generator = OllamaGenerator(
        settings,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "done": True,
                    "done_reason": "length",
                    "message": {"content": '{"sufficient":true,'},
                    "prompt_eval_count": 371,
                    "eval_count": 12,
                },
            )
        ),
    )
    budget = CallBudget(settings, generator, lambda: None, trace=RagTrace("m", "j"))
    with pytest.raises(LLMError, match="llm_output_limit"):
        budget.call([{"role": "user", "content": "Test"}], {}, 256)
    response = next(e for e in events(caplog) if e["event"] == "llm_response")
    assert response["response"] == '{"sufficient":true,'
    assert response["prompt_eval_count"] == 371
    assert response["eval_count"] == 12
    assert response["finish_reason"] == "length"
    assert budget.usage()["input_tokens"] == 371
    assert events(caplog)[-1]["reason"] == "llm_output_limit"


def test_missing_ollama_token_metrics_remain_unknown(caplog):
    caplog.set_level(logging.INFO, logger=logger.name)
    settings = Settings(_env_file=None)
    generator = OllamaGenerator(
        settings,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"done": True, "message": {"content": "{}"}})
        ),
    )
    budget = CallBudget(settings, generator, lambda: None, trace=RagTrace("m", "j"))
    budget.call([], {}, 256)
    response = next(e for e in events(caplog) if e["event"] == "llm_response")
    assert response["prompt_eval_count"] is None
    assert response["eval_count"] is None


def test_jsonl_file_survives_restart_escapes_control_characters_and_rotates(tmp_path):
    previous_handlers = logger.handlers.copy()
    previous_level = logger.level
    response = 'Тест\nПоддельная строка логов\r\t"цитата"'
    with rag_log_file(tmp_path):
        logger.handlers[-1].maxBytes = 500
        for i in range(5):
            RagTrace(f"m{i}", "j").emit("llm_response", response=response)
    path = tmp_path / "logs" / "rag.jsonl"
    assert logger.handlers == previous_handlers and logger.level == previous_level
    rotated = list(path.parent.glob("rag.jsonl*"))
    assert 1 < len(rotated) <= 4
    for file in rotated:
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        assert all(
            json.loads(line)["response"] == response for line in file.read_text().splitlines()
        )
    count = len(path.read_text().splitlines())
    with rag_log_file(tmp_path):
        RagTrace("last", "j").emit("answer_finished", status="completed")
    assert len(path.read_text().splitlines()) == count + 1
    assert json.loads(path.read_text().splitlines()[-1])["message_id"] == "last"


def test_budget_uses_whole_paragraph_fallback_and_marks_partial_article(caplog):
    caplog.set_level(logging.INFO, logger=logger.name)
    small = dict(
        source(1, "Тестовое правило действует при выполнении тестового условия."),
        heading="Статья 7. Тест",
        end=60,
        context_partial=True,
    )
    large = dict(
        source(1, "Очень длинная синтетическая статья. " * 2000),
        heading="Статья 7. Тест",
        fallback=small,
    )
    _, contexts = build_context(
        Settings(_env_file=None), "Вопрос", [], [large], trace=RagTrace("m", "j")
    )
    assert contexts["S1"]["text"] == small["text"]
    assert contexts["S1"]["context_partial"] is True
    assert events(caplog)[-1]["trimmed"][0]["reason"] == "paragraph_boundary"


def test_budget_marks_missing_continuation_instead_of_claiming_complete_article():
    first = dict(
        source(1, "Синтетическое правило действует при выполнении условия."),
        heading="Статья 7. Тест",
    )
    continuation = dict(source(2, "Синтетические исключения. " * 2000), heading="Статья 7. Тест")
    _, contexts = build_context(Settings(_env_file=None), "Вопрос", [], [first, continuation])
    assert list(contexts) == ["S1"]
    assert contexts["S1"]["context_partial"] is True

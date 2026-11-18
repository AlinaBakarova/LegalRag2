import pytest

from legal_workspace.retrieval.evidence import (
    article_key,
    query_terms,
    retrieval_query,
    structural_only,
)


@pytest.mark.parametrize(
    "text",
    [
        "Кодекс Российской Федерации об административных правонарушениях",
        "Глава 3. Административное наказание",
        "Статья 3.5. Административный штраф",
        "Раздел I. Общая часть\nГлава 3. Административное наказание",
    ],
)
def test_navigation_is_not_evidence(text):
    assert structural_only(text, text)


@pytest.mark.parametrize(
    "text",
    [
        "Статья 3.5. Административный штраф является денежным взысканием. Синтетический тест.",
        "Неустойка составляет 17 рублей.",
        "Срок оплаты — 17 дней.",
        "Статья 1. Тест\nЕсли поставка просрочена, платёж откладывается.",
    ],
)
def test_short_substantive_text_is_preserved(text):
    assert not structural_only(text, text)


def test_independent_question_does_not_inherit_previous_topic():
    history = [{"role": "user", "content": "Расскажи за что можно получить штраф"}]
    question = "Что такое административный штраф по КоАП РФ?"
    assert retrieval_query(question, history) == question
    assert query_terms(question) == ["административн", "штраф"]
    assert "штраф" in retrieval_query("А если оплатить позже?", history)
    assert article_key("Статья 3.5. Административный штраф") == "3.5"
    assert article_key("Глава 3. Общие положения") is None

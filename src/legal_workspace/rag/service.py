import json
import logging
import re
import time
from uuid import uuid4

from sqlalchemy import delete, func, select, update

from legal_workspace.llm.ollama import LLMError, OllamaGenerator
from legal_workspace.persistence.models import (
    ChatCitation,
    ChatDependency,
    ChatMessage,
    ChatSession,
    Chunk,
    Document,
    Job,
    SourceSpan,
)
from legal_workspace.rag.pipeline import CallBudget, build_context, generate_answer
from legal_workspace.rag.trace import RagTrace
from legal_workspace.retrieval.chunking import HEADING
from legal_workspace.retrieval.embeddings import SearchError
from legal_workspace.retrieval.evidence import ACTS, article_key, retrieval_query, structural_only
from legal_workspace.retrieval.service import SearchService

logger = logging.getLogger(__name__)
TERMINAL = {"completed", "insufficient", "failed", "removed", "cancelled"}
INSUFFICIENT = (
    "В доступных источниках недостаточно информации для обоснованного "
    "ответа. Уточните вопрос или добавьте документы."
)
ERRORS = {
    "llm_unavailable": "Не удалось подключиться к Ollama. Проверьте, что Ollama запущена.",
    "llm_model_missing": (
        "Модель не найдена в Ollama. Загрузите настроенную модель и повторите запрос."
    ),
    "llm_timeout": "Модель не успела ответить. Попробуйте задать более короткий вопрос.",
    "llm_api_error": "Ollama отклонила запрос. Проверьте её версию и выбранную модель.",
    "llm_invalid_response": "Не удалось прочитать ответ модели. Попробуйте повторить запрос.",
    "llm_busy": "Локальная модель сейчас недоступна или занята. Повторите запрос позже.",
    "model_not_ready": "Поисковая модель ещё не подготовлена.",
    "index_not_ready": "Документы ещё готовятся к поиску. Повторите вопрос после обработки.",
    "llm_context_limit": (
        "Вопрос и источники слишком велики для контекста модели. Сократите "
        "вопрос или выберите меньше документов."
    ),
    "llm_output_limit": "Модель не смогла завершить ответ. Попробуйте уточнить вопрос.",
    "llm_budget_exceeded": (
        "Достигнут лимит времени или попыток ответа. Уточните вопрос и повторите запрос."
    ),
    "invalid_citations": (
        "Не удалось подтвердить цитаты в ответе модели. Уточните вопрос и попробуйте снова."
    ),
    "unsupported_answer": (
        "Не удалось подтвердить ответ по источникам. Уточните вопрос и попробуйте снова."
    ),
    "sources_changed": "Источники изменились во время ответа. Повторите вопрос.",
    "processing_interrupted": "Формирование ответа было прервано. Повторите запрос.",
}


class ChatService:
    def __init__(self, database, *, search=None, generator=None):
        self.db = database
        self.settings = database.settings
        self.search = search if search is not None else SearchService(database)
        self.generator = generator if generator is not None else OllamaGenerator(self.settings)

    def sessions(self):
        with self.db.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(ChatSession.__table__).order_by(ChatSession.created_at.desc())
                ).mappings()
            ]

    def create_session(self):
        session_id = uuid4().hex
        with self.db.transaction() as connection:
            connection.execute(
                ChatSession.__table__.insert().values(
                    id=session_id, title="Новый диалог", created_at=time.time()
                )
            )
        return session_id

    def send(self, session_id, question, document_ids=None):
        question = question.strip()
        if not question or len(question) > 1500:
            raise LLMError("invalid_question")
        with self.db.transaction(immediate=True) as connection:
            if not connection.scalar(select(ChatSession.id).where(ChatSession.id == session_id)):
                raise LLMError("chat_not_found")
            if connection.scalar(
                select(ChatMessage.id).where(
                    ChatMessage.session_id == session_id,
                    ChatMessage.status.in_(["queued", "running"]),
                )
            ):
                raise LLMError("chat_busy")
            ordinal = connection.scalar(
                select(func.max(ChatMessage.ordinal)).where(ChatMessage.session_id == session_id)
            )
            ordinal = 0 if ordinal is None else ordinal + 1
            user_id, reply_id = uuid4().hex, uuid4().hex
            now = time.time()
            connection.execute(
                ChatMessage.__table__.insert(),
                [
                    dict(
                        id=user_id,
                        session_id=session_id,
                        ordinal=ordinal,
                        role="user",
                        content=question,
                        status="completed",
                        reply_to=None,
                        scope=document_ids,
                        usage={},
                        created_at=now,
                    ),
                    dict(
                        id=reply_id,
                        session_id=session_id,
                        ordinal=ordinal + 1,
                        role="assistant",
                        content="",
                        status="queued",
                        reply_to=user_id,
                        scope=document_ids,
                        usage={},
                        created_at=now,
                    ),
                ],
            )
            if ordinal == 0:
                connection.execute(
                    update(ChatSession)
                    .where(ChatSession.id == session_id)
                    .values(title=question[:70])
                )
            self._enqueue(connection, reply_id)
        return reply_id

    @staticmethod
    def _enqueue(connection, reply_id):
        now = time.time()
        connection.execute(
            Job.__table__.insert().values(
                id=uuid4().hex,
                kind="chat.answer",
                payload={"message_id": reply_id},
                status="queued",
                priority=10,
                attempts=0,
                max_attempts=1,
                created_at=now,
                updated_at=now,
                available_at=now,
            )
        )

    def reconcile(self):
        with self.db.transaction() as connection:
            failed = connection.execute(
                select(Job.payload, Job.error_code).where(
                    Job.kind == "chat.answer", Job.status == "failed"
                )
            ).all()
            for payload, error in failed:
                connection.execute(
                    update(ChatMessage)
                    .where(
                        ChatMessage.id == payload.get("message_id"),
                        ChatMessage.status.in_(["queued", "running"]),
                    )
                    .values(
                        status="failed", content="", error_code=error or "processing_interrupted"
                    )
                )

    def messages(self, session_id):
        self.reconcile()
        with self.db.engine.connect() as connection:
            messages = [
                dict(row)
                for row in connection.execute(
                    select(ChatMessage.__table__)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.ordinal)
                ).mappings()
            ]
            for message in messages:
                message["citations"] = [
                    dict(row)
                    for row in connection.execute(
                        select(ChatCitation.__table__).where(
                            ChatCitation.message_id == message["id"]
                        )
                    ).mappings()
                ]
                if message["status"] == "failed":
                    message["error"] = ERRORS.get(
                        message["error_code"], "Не удалось сформировать ответ. Повторите запрос."
                    )
                message.pop("error_code")
        return messages

    def retry(self, reply_id):
        with self.db.transaction(immediate=True) as connection:
            row = (
                connection.execute(select(ChatMessage.__table__).where(ChatMessage.id == reply_id))
                .mappings()
                .first()
            )
            if (
                row is None
                or row["role"] != "assistant"
                or row["status"] not in {"failed", "cancelled"}
            ):
                raise LLMError("invalid_retry")
            if connection.scalar(
                select(ChatMessage.id).where(
                    ChatMessage.session_id == row["session_id"],
                    ChatMessage.status.in_(["queued", "running"]),
                )
            ):
                raise LLMError("chat_busy")
            self._delete_jobs(connection, {reply_id})
            connection.execute(delete(ChatCitation).where(ChatCitation.message_id == reply_id))
            connection.execute(delete(ChatDependency).where(ChatDependency.message_id == reply_id))
            connection.execute(
                update(ChatMessage)
                .where(ChatMessage.id == reply_id)
                .values(status="queued", content="", error_code=None, usage={})
            )
            self._enqueue(connection, reply_id)

    @staticmethod
    def _delete_jobs(connection, message_ids):
        for job_id, payload in connection.execute(
            select(Job.id, Job.payload).where(Job.kind == "chat.answer")
        ).all():
            if payload.get("message_id") in message_ids:
                connection.execute(delete(Job).where(Job.id == job_id))

    def cancel(self, reply_id):
        with self.db.transaction(immediate=True) as connection:
            self._delete_jobs(connection, {reply_id})
            connection.execute(
                update(ChatMessage)
                .where(ChatMessage.id == reply_id, ChatMessage.status.in_(["queued", "running"]))
                .values(status="cancelled", content="")
            )

    def delete_session(self, session_id):
        with self.db.transaction(immediate=True) as connection:
            ids = set(
                connection.execute(
                    select(ChatMessage.id).where(ChatMessage.session_id == session_id)
                ).scalars()
            )
            self._delete_jobs(connection, ids)
            connection.execute(delete(ChatSession).where(ChatSession.id == session_id))

    @staticmethod
    def invalidate_document(connection, document_id):
        ids = set(
            connection.execute(
                select(ChatDependency.message_id).where(ChatDependency.document_id == document_id)
            ).scalars()
        )
        if not ids:
            return
        ChatService._delete_jobs(connection, ids)
        connection.execute(delete(ChatCitation).where(ChatCitation.message_id.in_(ids)))
        connection.execute(delete(ChatDependency).where(ChatDependency.message_id.in_(ids)))
        connection.execute(
            update(ChatMessage)
            .where(ChatMessage.id.in_(ids))
            .values(
                status="removed",
                content="",
                error_code=None,
                usage={},
            )
        )

    @staticmethod
    def _owned(connection, claim):
        return bool(
            connection.scalar(
                select(Job.id)
                .join(ChatMessage, ChatMessage.id == claim.payload["message_id"])
                .where(
                    Job.id == claim.id,
                    Job.lease_token == claim.token,
                    Job.status == "running",
                    Job.lease_expires_at > time.time(),
                    ChatMessage.status.in_(["queued", "running"]),
                )
            )
        )

    def _live(self, connection, claim):
        reply_id = claim.payload["message_id"]
        live = connection.scalar(
            select(Job.id).where(
                Job.id == claim.id,
                Job.lease_token == claim.token,
                Job.status == "running",
                Job.lease_expires_at > time.time(),
            )
        )
        reply = connection.scalar(
            select(ChatMessage.id).where(
                ChatMessage.id == reply_id, ChatMessage.status.in_(["queued", "running"])
            )
        )
        if not live or not reply:
            raise LLMError("sources_changed")
        for document_id, revision in connection.execute(
            select(ChatDependency.document_id, ChatDependency.text_revision).where(
                ChatDependency.message_id == reply_id
            )
        ):
            if not connection.scalar(
                select(Document.id).where(
                    Document.id == document_id,
                    Document.text_revision == revision,
                    Document.status.in_(["ready", "needs_review"]),
                )
            ):
                raise LLMError("sources_changed")

    def _check_live(self, claim):
        with self.db.engine.connect() as connection:
            self._live(connection, claim)

    def _history(self, connection, reply):
        prior = (
            connection.execute(
                select(ChatMessage.__table__)
                .where(
                    ChatMessage.session_id == reply["session_id"],
                    ChatMessage.ordinal < reply["ordinal"] - 1,
                    ChatMessage.status == "completed",
                )
                .order_by(ChatMessage.ordinal.desc())
                .limit(6)
            )
            .mappings()
            .all()
        )
        prior = list(reversed(prior))
        dependencies = {}
        for message in prior:
            for document_id, revision in connection.execute(
                select(ChatDependency.document_id, ChatDependency.text_revision).where(
                    ChatDependency.message_id == message["id"]
                )
            ):
                if document_id in dependencies and dependencies[document_id] != revision:
                    raise LLMError("sources_changed")
                dependencies[document_id] = revision
        return [{"role": m["role"], "content": m["content"]} for m in prior], dependencies

    def _question_scope(self, question, scope, trace):
        if scope is not None:
            return scope
        acts = [key for key, pattern in ACTS.items() if pattern.search(question)]
        # Comparisons must keep user documents available alongside legislation.
        comparison = any(word in question.casefold() for word in ("договор", "документ", "сравн"))
        if not acts or comparison:
            return scope
        with self.db.engine.connect() as connection:
            documents = connection.execute(
                select(Document.id, Document.filename, Document.source, Document.corpus_key).where(
                    Document.kind == "base", Document.status.in_(["ready", "needs_review"])
                )
            ).all()
        matched = []
        for doc in documents:
            title = doc.source.get("title", "") + " " + doc.filename
            key = (doc.corpus_key or "").casefold()
            if any(
                ACTS[act].search(title)
                or key == {"koap": "koap-rf", "uk": "uk-rf", "gk": "gk-rf"}[act]
                for act in acts
            ):
                matched.append(doc.id)
        if trace:
            trace.emit("retrieval_scope", requested_acts=acts, document_ids=matched)
        return matched

    def _expanded_source(self, item):
        # Enclose the hit in its article's region on this page, preserving source offsets.
        with self.db.engine.connect() as connection:
            text = connection.scalar(
                select(SourceSpan.text).where(SourceSpan.id == item["span_id"])
            )
            if text is None:
                raise LLMError("sources_changed")
            boundaries = (
                [0] + [m.start() for m in HEADING.finditer(text) if m.start() > 0] + [len(text)]
            )
            left = max(bound for bound in boundaries[:-1] if bound <= item["start"])
            right = next(bound for bound in boundaries[1:] if bound > item["start"])
            # Paragraph boundaries first; complete sentences for unstructured prose.
            edges = [0] + [m.end() for m in re.finditer(r"\n\s*\n", text)] + [len(text)]
            if len(edges) == 2 and not article_key(item["heading"]):
                edges = (
                    [0]
                    + [m.end() for m in re.finditer(r"[.!?]\s+(?=[А-ЯЁA-Z])", text)]
                    + [len(text)]
                )
            small_left = max(left, max(e for e in edges if e <= item["start"]))
            small_right = min(right, next((e for e in edges if e >= item["end"]), len(text)))
            chunks = connection.execute(
                select(Chunk.id, Chunk.start, Chunk.end)
                .where(
                    Chunk.generation_id == item["generation_id"],
                    Chunk.span_id == item["span_id"],
                    Chunk.start < right,
                    Chunk.end > left,
                )
                .order_by(Chunk.ordinal)
            ).all()
        fallback = dict(
            item,
            text=text[small_left:small_right],
            start=small_left,
            end=small_right,
            chunk_ids=[r.id for r in chunks if r.start < small_right and r.end > small_left],
            context_partial=(small_left > left or small_right < right),
        )
        if not article_key(item["heading"]):
            return fallback
        return dict(
            item,
            text=text[left:right],
            start=left,
            end=right,
            chunk_ids=[r.id for r in chunks],
            fallback=fallback,
        )

    def _context_sources(self, question, scope, history, *, trace=None):
        query = retrieval_query(question, history)
        scope = self._question_scope(query, scope, trace)
        found = self.search.search(query, document_ids=scope, limit=20, evidence=True, trace=trace)
        if trace:
            trace.emit(
                "retrieval_results",
                chunk_ids=[item["id"] for item in found],
                history_used=query != question,
            )
        anchors, continuations, seen = [], [], set()
        discarded, duplicates, limited = [], [], set()

        def add(item, target):
            if structural_only(item["text"], item["heading"]):
                discarded.append(item["id"])
                return
            expanded = self._expanded_source(item)
            key = (expanded["span_id"], expanded["start"], expanded["end"])
            if key not in seen:
                seen.add(key)
                target.append(expanded)
            else:
                duplicates.append(item["id"])

        # All independent hits are considered before optional article continuations.
        for item in found:
            add(item, anchors)
        articles = set()
        for item in found:
            key = (item["generation_id"], article_key(item["heading"]))
            if not key[1] or key in articles:
                continue
            articles.add(key)
            with self.db.engine.connect() as connection:
                ids = (
                    connection.execute(
                        select(Chunk.id)
                        .where(
                            Chunk.generation_id == item["generation_id"],
                            Chunk.heading == item["heading"],
                        )
                        .order_by(Chunk.ordinal)
                        .limit(65)
                    )
                    .scalars()
                    .all()
                )
            if len(ids) > 64:
                limited.add(key)
            for chunk_id in ids[:64]:
                add(self.search.source(chunk_id), continuations)
        for source in anchors + continuations:
            if (source["generation_id"], article_key(source["heading"])) in limited:
                source["context_partial"] = True
        if trace:
            trace.emit(
                "context_candidates",
                added_neighbour_chunk_ids=[item["id"] for item in continuations],
                duplicate_chunk_ids=list(dict.fromkeys(duplicates)),
                candidate_chunk_ids=[item["id"] for item in anchors + continuations],
                dropped_before_build_context=[
                    {"chunk_id": cid, "reason": "structural_only"}
                    for cid in dict.fromkeys(discarded)
                ],
                expansion="article_regions_anchors_first",
            )
        return anchors + continuations

    @staticmethod
    def _verify_proof(connection, citation):
        row = connection.execute(
            select(SourceSpan.text, Document.text_revision, Document.status, Document.id)
            .join(Document, Document.id == SourceSpan.document_id)
            .where(SourceSpan.id == citation["span_id"], Document.id == citation["document_id"])
        ).first()
        if (
            row is None
            or row.text_revision != citation["text_revision"]
            or row.status not in {"ready", "needs_review"}
            or row.text[citation["start"] : citation["end"]] != citation["quote"]
        ):
            raise LLMError("sources_changed")

    def citation(self, citation_id):
        with self.db.engine.connect() as connection:
            row = (
                connection.execute(
                    select(ChatCitation.__table__)
                    .join(ChatMessage, ChatMessage.id == ChatCitation.message_id)
                    .where(ChatCitation.id == citation_id, ChatMessage.status == "completed")
                )
                .mappings()
                .first()
            )
            if row is None:
                raise LLMError("citation_not_found")
            self._verify_proof(connection, row)
            return dict(row)

    def process(self, claim):
        reply_id = claim.payload["message_id"]
        trace = RagTrace(reply_id, claim.id)
        budget = CallBudget(
            self.settings, self.generator, lambda: self._check_live(claim), trace=trace
        )
        coverage = {}
        try:
            with self.db.transaction(immediate=True) as connection:
                reply = (
                    connection.execute(
                        select(ChatMessage.__table__).where(ChatMessage.id == reply_id)
                    )
                    .mappings()
                    .first()
                )
                if reply is None or reply["status"] in TERMINAL:
                    trace.emit("answer_skipped", reason="missing_or_terminal")
                    return {"skipped": True}
                trace.question_id = reply["reply_to"]
                trace.emit("answer_started", model=self.settings.llm_model)
                self._live(connection, claim)
                question = connection.scalar(
                    select(ChatMessage.content).where(ChatMessage.id == reply["reply_to"])
                )
                history, dependencies = self._history(connection, reply)
                connection.execute(
                    update(ChatMessage).where(ChatMessage.id == reply_id).values(status="running")
                )
            trace.stage = "retrieval"
            sources = self._context_sources(question, reply["scope"], history, trace=trace)
            if sources:
                trace.stage = "build_context"
                counter_factory = getattr(self.generator, "token_counter", None)
                if counter_factory is not None:
                    budget.counter = counter_factory()
                messages, contexts = build_context(
                    self.settings, question, history, sources, trace=trace, counter=budget.counter
                )
                search_status = self.search.status()
                coverage = {
                    "partial_sources": any(s["warnings"] for s in contexts.values()),
                    "partial_context": any(s.get("context_partial") for s in contexts.values()),
                    "partial_corpus": search_status["indexed"] < search_status["available"],
                }
                for source in contexts.values():
                    if (
                        source["document_id"] in dependencies
                        and dependencies[source["document_id"]] != source["text_revision"]
                    ):
                        raise LLMError("sources_changed")
                    dependencies[source["document_id"]] = source["text_revision"]
            else:
                contexts = {}
                trace.emit("answer_abstained", reason="no_sources")
            trace.stage = "source_validation"
            with self.db.transaction(immediate=True) as connection:
                self._live(connection, claim)
                connection.execute(
                    delete(ChatDependency).where(ChatDependency.message_id == reply_id)
                )
                for document_id, revision in dependencies.items():
                    if not connection.scalar(
                        select(Document.id).where(
                            Document.id == document_id,
                            Document.text_revision == revision,
                            Document.status.in_(["ready", "needs_review"]),
                        )
                    ):
                        raise LLMError("sources_changed")
                    connection.execute(
                        ChatDependency.__table__.insert().values(
                            message_id=reply_id, document_id=document_id, text_revision=revision
                        )
                    )
            if contexts:
                draft, citations = generate_answer(budget, messages, contexts, question)
                content = (
                    json.dumps([c.text for c in draft.claims], ensure_ascii=False)
                    if draft.sufficient
                    else INSUFFICIENT
                )
                status = "completed" if draft.sufficient else "insufficient"
            else:
                content, status, citations = INSUFFICIENT, "insufficient", []
            trace.stage = "publication"
            with self.db.transaction(immediate=True) as connection:
                self._live(connection, claim)
                for citation in citations:
                    self._verify_proof(connection, citation)
                    connection.execute(
                        ChatCitation.__table__.insert().values(
                            id=uuid4().hex, message_id=reply_id, **citation
                        )
                    )
                connection.execute(
                    update(ChatMessage)
                    .where(ChatMessage.id == reply_id)
                    .values(
                        status=status,
                        content=content,
                        usage={**budget.usage(), **coverage},
                        model=self.settings.llm_model,
                        error_code=None,
                    )
                )
            trace.emit("answer_finished", status=status, **budget.usage())
            logger.info(
                "chat_answer_finished message_id=%s status=%s calls=%s",
                reply_id,
                status,
                budget.calls,
            )
            return {"answered": True}
        except (LLMError, SearchError) as exc:
            code = str(exc)
            trace.emit("answer_failed", reason=code, **budget.usage())
            with self.db.transaction(immediate=True) as connection:
                if not self._owned(connection, claim):
                    trace.emit("answer_discarded", reason="not_owned_or_sources_changed")
                    return {"skipped": True}
                connection.execute(
                    update(ChatMessage)
                    .where(ChatMessage.id == reply_id)
                    .values(
                        status="failed",
                        content="",
                        error_code=code,
                        usage={**budget.usage(), **coverage},
                        model=self.settings.llm_model,
                    )
                )
            logger.warning("chat_answer_failed message_id=%s code=%s", reply_id, code)
            return {"status": "failed", "error_code": code}
        except Exception as exc:
            trace.emit("answer_failed", reason="unexpected_error", error_type=type(exc).__name__)
            raise

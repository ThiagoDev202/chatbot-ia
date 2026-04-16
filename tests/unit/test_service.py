"""Testes unitários de ``ChatService`` usando ``ConversationStore`` real e LLM fake."""

from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import ValidationError

from app.chat.chain import build_chat_chain
from app.chat.memory import ConversationStore
from app.chat.schemas import ChatRequest
from app.chat.service import ChatService
from app.core.config import Settings
from app.core.exceptions import LLMUnavailableError, SessionNotFoundError


def _make_history_factory(
    histories: dict[str, InMemoryChatMessageHistory],
) -> Callable[[str], BaseChatMessageHistory]:
    """Factory síncrona — o Langchain chama assim — apoiada num dict externo."""

    def _factory(session_id: str) -> BaseChatMessageHistory:
        if session_id not in histories:
            histories[session_id] = InMemoryChatMessageHistory()
        return histories[session_id]

    return _factory


def _make_service(
    responses: list[str] | None = None,
    model_name: str = "gpt-4o-mini-test",
) -> ChatService:
    """Constrói um ``ChatService`` com store real, chain real e ``FakeListChatModel``.

    Compartilha o dicionário de históricos entre o ``ConversationStore`` (que expõe
    ``get``/``reset``) e o ``history_factory`` (que alimenta o chain), de modo que
    o serviço e o chain enxerguem a mesma sessão.
    """
    if responses is None:
        responses = ["Resposta de teste"]

    histories: dict[str, InMemoryChatMessageHistory] = {}
    store = ConversationStore()
    # Redireciona o dicionário interno do store para compartilhar com o factory.
    # Isso permite que ``get_or_create``/``get``/``reset`` do store e o
    # ``history_factory`` do chain apontem para a mesma estrutura.
    store._store = histories  # type: ignore[attr-defined]

    settings = Settings(openai_api_key="sk-fake")
    fake_llm = FakeListChatModel(responses=responses)
    chain = build_chat_chain(settings, _make_history_factory(histories), llm=fake_llm)
    return ChatService(store=store, chain=chain, model_name=model_name)


class _ExplodingChain(Runnable[dict[str, Any], str]):
    """Runnable fake que sempre levanta ``RuntimeError`` para exercitar o handler de erro."""

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("boom")

    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("boom")


async def test_ask_generates_session_id_when_none() -> None:
    """Sem ``session_id``, o serviço deve gerar um UUID válido."""
    service = _make_service()
    response = await service.ask(None, "Oi")

    assert response.session_id
    # Deve parsear como UUID (levanta ``ValueError`` caso contrário).
    UUID(response.session_id)


async def test_ask_generates_session_id_when_empty_string() -> None:
    """``session_id`` vazio deve ser tratado como ausente e gerar novo UUID."""
    service = _make_service()
    response = await service.ask("", "Oi")

    assert response.session_id
    UUID(response.session_id)


async def test_ask_uses_provided_session_id() -> None:
    """Quando o caller envia um ``session_id``, ele deve ser preservado."""
    service = _make_service()
    response = await service.ask("sid-custom", "Oi")
    assert response.session_id == "sid-custom"


async def test_ask_returns_llm_response_as_answer() -> None:
    """O campo ``answer`` deve conter exatamente a string produzida pelo LLM fake."""
    service = _make_service(responses=["Saída exata do tutor."])
    response = await service.ask("sid-1", "Pergunta")
    assert response.answer == "Saída exata do tutor."


async def test_ask_populates_model_name() -> None:
    """``response.model`` deve refletir o ``model_name`` injetado no service."""
    service = _make_service(model_name="gpt-test-42")
    response = await service.ask("sid-1", "Pergunta")
    assert response.model == "gpt-test-42"


async def test_ask_has_timezone_aware_created_at() -> None:
    """``created_at`` deve carregar timezone (UTC)."""
    service = _make_service()
    response = await service.ask("sid-1", "Pergunta")
    assert response.created_at.tzinfo is not None


async def test_multi_turn_preserves_context() -> None:
    """Duas chamadas com o mesmo ``session_id`` devem acumular 4 mensagens no histórico."""
    service = _make_service(responses=["Primeira resposta.", "Segunda resposta."])
    session_id = "sid-multi"

    first = await service.ask(session_id, "Primeira pergunta?")
    second = await service.ask(session_id, "Segunda pergunta?")

    assert first.answer == "Primeira resposta."
    assert second.answer == "Segunda resposta."

    history = await service.get_history(session_id)
    assert len(history.messages) == 4

    assert history.messages[0].role == "user"
    assert history.messages[0].content == "Primeira pergunta?"
    assert history.messages[1].role == "assistant"
    assert history.messages[1].content == "Primeira resposta."
    assert history.messages[2].role == "user"
    assert history.messages[2].content == "Segunda pergunta?"
    assert history.messages[3].role == "assistant"
    assert history.messages[3].content == "Segunda resposta."


async def test_ask_raises_llm_unavailable_on_chain_error() -> None:
    """Qualquer exceção do chain deve virar ``LLMUnavailableError``."""
    store = ConversationStore()
    service = ChatService(
        store=store,
        chain=_ExplodingChain(),
        model_name="gpt-4o-mini-test",
    )

    with pytest.raises(LLMUnavailableError) as exc_info:
        await service.ask("sid-err", "Pergunta")

    assert "LLM indisponível" in str(exc_info.value)


async def test_get_history_raises_for_unknown_session() -> None:
    """``get_history`` deve propagar ``SessionNotFoundError`` para sessão inexistente."""
    service = _make_service()
    with pytest.raises(SessionNotFoundError):
        await service.get_history("inexistente")


async def test_get_history_returns_user_and_assistant_messages() -> None:
    """Após um ``ask``, o histórico deve conter duas mensagens com roles corretos."""
    service = _make_service(responses=["Resposta única."])
    session_id = "sid-hist"

    await service.ask(session_id, "Como criar uma lista?")
    history = await service.get_history(session_id)

    assert history.session_id == session_id
    assert len(history.messages) == 2
    assert history.messages[0].role == "user"
    assert history.messages[0].content == "Como criar uma lista?"
    assert history.messages[1].role == "assistant"
    assert history.messages[1].content == "Resposta única."
    assert history.messages[0].created_at.tzinfo is not None
    assert history.messages[1].created_at.tzinfo is not None


async def test_reset_clears_session() -> None:
    """Após ``reset``, ``get_history`` deve levantar ``SessionNotFoundError``."""
    service = _make_service()
    session_id = "sid-reset"

    await service.ask(session_id, "Pergunta")
    await service.reset(session_id)

    with pytest.raises(SessionNotFoundError):
        await service.get_history(session_id)


async def test_reset_raises_for_unknown_session() -> None:
    """``reset`` sobre sessão inexistente deve levantar ``SessionNotFoundError``."""
    service = _make_service()
    with pytest.raises(SessionNotFoundError):
        await service.reset("inexistente")


async def test_get_history_skips_non_user_assistant_messages() -> None:
    """Mensagens de sistema (ou outros tipos) não devem aparecer no histórico exposto."""
    service = _make_service()
    session_id = "sid-mixed"

    # Injeta diretamente no histórico uma ``SystemMessage`` e uma ``AIMessage`` com
    # conteúdo não-string para cobrir os ramos de filtragem/conversão.
    history = await service._store.get_or_create(session_id)  # type: ignore[attr-defined]
    history.messages.append(SystemMessage(content="Instrução interna."))
    history.messages.append(HumanMessage(content="Pergunta."))
    history.messages.append(AIMessage(content=["Parte A", "Parte B"]))

    result = await service.get_history(session_id)

    # A ``SystemMessage`` deve ser ignorada; restam user + assistant.
    assert [m.role for m in result.messages] == ["user", "assistant"]
    assert result.messages[0].content == "Pergunta."
    # Conteúdo não-string foi convertido via ``str(...)``.
    assert "Parte A" in result.messages[1].content


def test_chat_request_rejects_whitespace_only_message() -> None:
    """``ChatRequest`` deve rejeitar mensagens apenas com espaços (após strip)."""
    # O ``str_strip_whitespace=True`` normaliza antes de ``min_length``, então
    # uma string só com espaços colide com ``min_length=1``.
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_chat_request_accepts_valid_message() -> None:
    """Um ``message`` não-branco deve percorrer o validador customizado e ser aceito."""
    request = ChatRequest(message="Como criar uma lista em Python?")
    assert request.message == "Como criar uma lista em Python?"
    assert request.session_id is None

"""Testes de integração end-to-end dos endpoints ``/api/v1/chat``."""

from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest
from httpx import AsyncClient
from langchain_core.runnables import Runnable

from app.chat.service import ChatService


async def test_post_chat_creates_session_and_returns_answer(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
) -> None:
    """POST sem session_id deve gerar um UUID válido e devolver a resposta fake."""
    chat_service_factory(responses=["Use colchetes: frutas = ['maca']."])

    response = await async_client.post(
        "/api/v1/chat",
        json={"message": "Como criar uma lista em Python?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use colchetes: frutas = ['maca']."
    assert body["model"] == "gpt-4o-mini-test"
    assert body["created_at"]

    session_id = body["session_id"]
    assert session_id
    # Deve parsear como UUID; ``ValueError`` caso contrário.
    UUID(session_id)


async def test_post_chat_with_existing_session_reuses_it(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
) -> None:
    """Dois POSTs com mesmo session_id devem preservar o id e usar respostas distintas."""
    chat_service_factory(responses=["Primeira.", "Segunda."])
    session_id = "sid-reusa"

    first = await async_client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "Pergunta 1"},
    )
    second = await async_client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "Pergunta 2"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["session_id"] == session_id
    assert second.json()["session_id"] == session_id
    assert first.json()["answer"] == "Primeira."
    assert second.json()["answer"] == "Segunda."


async def test_post_chat_preserves_context_across_turns(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
) -> None:
    """Dois turnos + GET devem resultar em 4 mensagens na ordem correta."""
    chat_service_factory(responses=["R1", "R2"])
    session_id = "sid-multi"

    await async_client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "P1"},
    )
    await async_client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "P2"},
    )

    history = await async_client.get(f"/api/v1/chat/{session_id}")

    assert history.status_code == 200
    body = history.json()
    assert body["session_id"] == session_id
    messages = body["messages"]
    assert len(messages) == 4
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in messages] == ["P1", "R1", "P2", "R2"]


async def test_post_chat_rejects_empty_message(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
) -> None:
    """POST com ``message=""`` deve retornar 422 (validação Pydantic)."""
    chat_service_factory()

    response = await async_client.post(
        "/api/v1/chat",
        json={"message": ""},
    )

    assert response.status_code in (400, 422)


async def test_post_chat_rejects_message_too_long(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
) -> None:
    """POST com ``message`` acima de 4000 chars deve retornar 422."""
    chat_service_factory()

    response = await async_client.post(
        "/api/v1/chat",
        json={"message": "a" * 4001},
    )

    assert response.status_code in (400, 422)


async def test_get_history_returns_404_for_unknown_session(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
) -> None:
    """GET em sessão inexistente deve responder 404 com detalhe em PT-BR."""
    chat_service_factory()

    response = await async_client.get("/api/v1/chat/sessao-fantasma")

    assert response.status_code == 404
    body = response.json()
    assert "sessao-fantasma" in body["detail"]
    assert "não encontrada" in body["detail"]


async def test_delete_session_returns_204(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
) -> None:
    """Criar sessão, deletar (204) e garantir que GET posterior responde 404."""
    chat_service_factory(responses=["Resposta única."])
    session_id = "sid-delete"

    create = await async_client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "Pergunta"},
    )
    assert create.status_code == 200

    delete = await async_client.delete(f"/api/v1/chat/{session_id}")
    assert delete.status_code == 204
    assert delete.content == b""

    follow_up = await async_client.get(f"/api/v1/chat/{session_id}")
    assert follow_up.status_code == 404


async def test_delete_unknown_session_returns_404(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
) -> None:
    """DELETE em sessão inexistente deve retornar 404 com mensagem em PT-BR."""
    chat_service_factory()

    response = await async_client.delete("/api/v1/chat/nao-existe")

    assert response.status_code == 404
    body = response.json()
    assert "nao-existe" in body["detail"]
    assert "não encontrada" in body["detail"]


async def test_post_chat_returns_503_on_llm_failure(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
    exploding_chain: Runnable[dict[str, Any], str],
) -> None:
    """Falha do chain deve virar 503 com detalhe descritivo em PT-BR."""
    chat_service_factory(chain=exploding_chain)

    response = await async_client.post(
        "/api/v1/chat",
        json={"session_id": "sid-boom", "message": "Pergunta qualquer"},
    )

    assert response.status_code == 503
    body = response.json()
    assert "LLM indisponível" in body["detail"]
    assert "OPENAI_API_KEY" in body["detail"]


async def test_session_not_found_on_post_uses_fallback_message(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SessionNotFoundError`` em rota sem ``session_id`` no path usa mensagem da exceção."""
    from app.core.exceptions import SessionNotFoundError

    service = chat_service_factory()

    async def _raise_not_found(self: ChatService, session_id: str | None, message: str) -> Any:
        raise SessionNotFoundError("Sessão 'xyz' não encontrada.")

    monkeypatch.setattr(ChatService, "ask", _raise_not_found)

    response = await async_client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "Sessão 'xyz' não encontrada."
    assert service is not None


async def test_session_not_found_without_message_uses_generic_detail(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem mensagem na exceção nem session_id no path, usa detalhe genérico."""
    from app.core.exceptions import SessionNotFoundError

    service = chat_service_factory()

    async def _raise_not_found(self: ChatService, session_id: str | None, message: str) -> Any:
        raise SessionNotFoundError()

    monkeypatch.setattr(ChatService, "ask", _raise_not_found)

    response = await async_client.post("/api/v1/chat", json={"message": "oi"})

    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "Sessão não encontrada."
    assert service is not None


async def test_get_chat_service_reuses_store_and_chain_singletons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_chat_service`` deve reaproveitar o store e o chain entre chamadas.

    Substitui ``build_chat_chain`` para não instanciar ``ChatOpenAI`` — o que
    exigiria ``OPENAI_API_KEY`` real no ambiente.
    """
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    import app.chat.router as router_module
    from app.chat.chain import build_chat_chain as real_build_chat_chain
    from app.core.config import Settings

    def _fake_build_chain(
        settings: Settings,
        history_factory: Any,
        llm: Any = None,
    ) -> Runnable[dict[str, Any], str]:
        return real_build_chat_chain(
            settings,
            history_factory=history_factory,
            llm=FakeListChatModel(responses=["stub"]),
        )

    monkeypatch.setattr(router_module, "build_chat_chain", _fake_build_chain)
    monkeypatch.setattr(router_module, "_store", None)
    monkeypatch.setattr(router_module, "_chain", None)

    service_a = router_module.get_chat_service()
    service_b = router_module.get_chat_service()

    # O service é recriado (barato), mas store e chain devem ser reaproveitados.
    assert service_a._store is service_b._store
    assert service_a._chain is service_b._chain
    assert isinstance(service_a, ChatService)

    # Limpeza: zera os singletons para não afetar testes subsequentes.
    monkeypatch.setattr(router_module, "_store", None)
    monkeypatch.setattr(router_module, "_chain", None)


async def test_post_chat_returns_503_when_openai_key_missing(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem ``OPENAI_API_KEY`` configurada, POST /chat deve responder 503 — não 500.

    Exercita o caminho real de construção do chain (sem override de dependência),
    garantindo que a falha do ``ChatOpenAI.__init__`` é traduzida em
    ``LLMUnavailableError`` pelo ``_NoOpChain`` antes da resposta HTTP.
    """
    import app.chat.router as router_module
    from app.core.config import get_settings
    from app.main import app as main_app

    main_app.dependency_overrides.pop(router_module.get_chat_service, None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(router_module, "_store", None)
    monkeypatch.setattr(router_module, "_chain", None)

    try:
        response = await async_client.post(
            "/api/v1/chat",
            json={"message": "oi"},
        )

        assert response.status_code == 503
        body = response.json()
        assert "LLM indisponível" in body["detail"]
        assert "OPENAI_API_KEY" in body["detail"]
    finally:
        monkeypatch.setattr(router_module, "_store", None)
        monkeypatch.setattr(router_module, "_chain", None)
        get_settings.cache_clear()


async def test_get_history_returns_404_without_openai_key(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET e DELETE devem funcionar mesmo sem ``OPENAI_API_KEY`` (não tocam o LLM)."""
    import app.chat.router as router_module
    from app.core.config import get_settings
    from app.main import app as main_app

    main_app.dependency_overrides.pop(router_module.get_chat_service, None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(router_module, "_store", None)
    monkeypatch.setattr(router_module, "_chain", None)

    try:
        get_response = await async_client.get("/api/v1/chat/sessao-inexistente")
        assert get_response.status_code == 404
        assert "sessao-inexistente" in get_response.json()["detail"]

        delete_response = await async_client.delete("/api/v1/chat/sessao-inexistente")
        assert delete_response.status_code == 404
    finally:
        monkeypatch.setattr(router_module, "_store", None)
        monkeypatch.setattr(router_module, "_chain", None)
        get_settings.cache_clear()


async def test_chatbot_error_base_class_returns_500(
    async_client: AsyncClient,
    chat_service_factory: Callable[..., ChatService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Erros de domínio não especializados devem cair no handler 500."""
    from app.core.exceptions import ChatbotError

    service = chat_service_factory()

    async def _raise_base(self: ChatService, session_id: str | None, message: str) -> Any:
        raise ChatbotError("algo inesperado")

    monkeypatch.setattr(ChatService, "ask", _raise_base)

    response = await async_client.post(
        "/api/v1/chat",
        json={"message": "ping"},
    )

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Erro interno do chatbot."
    # Evita warning de fixture não utilizada.
    assert service is not None

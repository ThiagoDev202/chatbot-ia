"""Fixtures compartilhadas para os testes."""

from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from app.chat.chain import build_chat_chain
from app.chat.memory import ConversationStore
from app.chat.router import get_chat_service
from app.chat.service import ChatService
from app.core.config import Settings
from app.main import app


@pytest.fixture
async def async_client() -> AsyncIterator[AsyncClient]:
    """Retorna cliente HTTP assincrono apontando para a aplicacao."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
def chat_service_factory() -> Callable[..., ChatService]:
    """Factory que produz um ``ChatService`` com LLM fake e o registra no app.

    Uso típico:

    ```python
    service = chat_service_factory(responses=["ola", "tchau"])
    ```

    O service resultante é automaticamente instalado via
    ``app.dependency_overrides[get_chat_service]`` e é limpo ao final do teste.
    """

    def _factory(
        responses: list[str] | None = None,
        model_name: str = "gpt-4o-mini-test",
        llm: BaseChatModel | None = None,
        chain: Runnable[dict[str, Any], str] | None = None,
    ) -> ChatService:
        effective_responses = responses if responses is not None else ["Resposta fake do tutor."]

        store = ConversationStore()
        settings = Settings(openai_api_key="sk-fake")

        if chain is None:
            fake_llm: BaseChatModel = (
                llm if llm is not None else FakeListChatModel(responses=effective_responses)
            )
            chain = build_chat_chain(
                settings,
                history_factory=store.get_or_create_sync,
                llm=fake_llm,
            )

        service = ChatService(store=store, chain=chain, model_name=model_name)
        app.dependency_overrides[get_chat_service] = lambda: service
        return service

    yield _factory

    # Teardown: remove qualquer override instalado pelo teste.
    app.dependency_overrides.pop(get_chat_service, None)


class _ExplodingChain(Runnable[dict[str, Any], str]):
    """Runnable fake que sempre levanta erro ao ser invocado."""

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("LLM falhou (simulado em teste).")

    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("LLM falhou (simulado em teste).")


@pytest.fixture
def exploding_chain() -> Runnable[dict[str, Any], str]:
    """Chain que sempre levanta ``RuntimeError`` — exercita o caminho 503."""
    return _ExplodingChain()

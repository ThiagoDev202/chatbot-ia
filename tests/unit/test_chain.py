"""Testes unitários da camada Chain usando LLMs fake (sem chamadas reais)."""

from typing import Any, ClassVar

import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from app.chat.chain import build_chat_chain
from app.chat.prompts import PYTHON_TUTOR_SYSTEM_PROMPT
from app.core.config import Settings


class SpyChatModel(FakeListChatModel):
    """Fake LLM que captura as mensagens recebidas para inspeção."""

    received: ClassVar[list[list[BaseMessage]]] = []

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        type(self).received.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def settings() -> Settings:
    """Settings com chave fake; o LLM será sempre injetado nos testes."""
    return Settings(openai_api_key="sk-fake")


@pytest.fixture
def histories() -> dict[str, InMemoryChatMessageHistory]:
    """Armazenamento local de históricos para cada teste."""
    return {}


@pytest.fixture
def history_factory(
    histories: dict[str, InMemoryChatMessageHistory],
) -> Any:
    """Factory que devolve (ou cria) um histórico in-memory por session_id."""

    def _factory(session_id: str) -> BaseChatMessageHistory:
        if session_id not in histories:
            histories[session_id] = InMemoryChatMessageHistory()
        return histories[session_id]

    return _factory


async def test_build_chain_uses_provided_llm(
    settings: Settings,
    history_factory: Any,
) -> None:
    """O LLM injetado deve ser usado e sua resposta devolvida como string."""
    fake_llm = FakeListChatModel(responses=["Resposta de teste do tutor Python."])

    chain = build_chat_chain(settings, history_factory, llm=fake_llm)

    answer = await chain.ainvoke(
        {"question": "Oi"},
        config={"configurable": {"session_id": "sid-1"}},
    )

    assert answer == "Resposta de teste do tutor Python."


async def test_chain_appends_turn_to_history(
    settings: Settings,
    history_factory: Any,
    histories: dict[str, InMemoryChatMessageHistory],
) -> None:
    """Após uma invocação, o histórico deve conter a pergunta e a resposta."""
    fake_llm = FakeListChatModel(responses=["Uma resposta."])

    chain = build_chat_chain(settings, history_factory, llm=fake_llm)

    await chain.ainvoke(
        {"question": "Como criar uma lista?"},
        config={"configurable": {"session_id": "sid-1"}},
    )

    messages = histories["sid-1"].messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Como criar uma lista?"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Uma resposta."


async def test_chain_uses_history_for_subsequent_turn(
    settings: Settings,
    history_factory: Any,
    histories: dict[str, InMemoryChatMessageHistory],
) -> None:
    """No segundo turno, o LLM deve receber o histórico acumulado."""
    histories["sid-1"] = InMemoryChatMessageHistory()
    histories["sid-1"].add_user_message("Pergunta anterior.")
    histories["sid-1"].add_ai_message("Resposta anterior.")

    SpyChatModel.received = []
    spy_llm = SpyChatModel(responses=["Segunda resposta."])

    chain = build_chat_chain(settings, history_factory, llm=spy_llm)

    answer = await chain.ainvoke(
        {"question": "Nova pergunta"},
        config={"configurable": {"session_id": "sid-1"}},
    )

    assert answer == "Segunda resposta."
    assert len(SpyChatModel.received) == 1
    sent_messages = SpyChatModel.received[0]
    contents = [m.content for m in sent_messages]
    assert "Pergunta anterior." in contents
    assert "Resposta anterior." in contents
    assert "Nova pergunta" in contents

    # O histórico deve conter 4 mensagens (2 anteriores + novo par).
    assert len(histories["sid-1"].messages) == 4


def test_build_chain_defaults_to_chat_openai(
    settings: Settings,
    history_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem ``llm`` explícito, o chain deve instanciar ``ChatOpenAI`` via settings."""
    captured: dict[str, Any] = {}
    original_init = ChatOpenAI.__init__

    def spy_init(self: ChatOpenAI, *args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(ChatOpenAI, "__init__", spy_init)

    chain = build_chat_chain(settings, history_factory)

    assert chain is not None
    assert captured["model_name"] == settings.openai_model
    assert captured["temperature"] == settings.openai_temperature
    assert captured["openai_api_key"] is not None


async def test_system_prompt_is_applied(
    settings: Settings,
    history_factory: Any,
) -> None:
    """A primeira mensagem enviada ao LLM deve ser o system prompt do tutor."""
    SpyChatModel.received = []
    spy_llm = SpyChatModel(responses=["Resposta."])

    chain = build_chat_chain(settings, history_factory, llm=spy_llm)

    await chain.ainvoke(
        {"question": "Qual é a sintaxe de um for em Python?"},
        config={"configurable": {"session_id": "sid-sys"}},
    )

    assert SpyChatModel.received, "O LLM fake deve ter recebido ao menos uma chamada."
    first_call = SpyChatModel.received[0]
    assert isinstance(first_call[0], SystemMessage)
    assert first_call[0].content == PYTHON_TUTOR_SYSTEM_PROMPT

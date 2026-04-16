"""Construção do pipeline Langchain utilizado pelo chatbot tutor de Python."""

from collections.abc import Callable
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.chat.prompts import PYTHON_TUTOR_SYSTEM_PROMPT
from app.core.config import Settings


def build_chat_chain(
    settings: Settings,
    history_factory: Callable[[str], BaseChatMessageHistory],
    llm: BaseChatModel | None = None,
) -> Runnable[dict[str, Any], str]:
    """Monta o pipeline de chat (prompt + LLM + parser) com histórico por sessão.

    Em testes, injete um ``llm`` fake para evitar chamadas reais à OpenAI.
    """
    # Usar ``SystemMessage`` direto (em vez de tupla ``("system", ...)``) evita
    # que o ChatPromptTemplate interprete chaves ``{topico}`` do prompt literal
    # (em PT-BR) como variáveis de template.
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=PYTHON_TUTOR_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chat_model: BaseChatModel = llm if llm is not None else _build_default_llm(settings)

    core_chain: Runnable[dict[str, Any], str] = prompt | chat_model | StrOutputParser()

    chain_with_history: Runnable[dict[str, Any], str] = RunnableWithMessageHistory(
        core_chain,
        get_session_history=history_factory,
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history


def _build_default_llm(settings: Settings) -> ChatOpenAI:
    """Instancia o ``ChatOpenAI`` padrão a partir das configurações da aplicação."""
    api_key: SecretStr | None = (
        SecretStr(settings.openai_api_key) if settings.openai_api_key is not None else None
    )
    return ChatOpenAI(
        model_name=settings.openai_model,
        temperature=settings.openai_temperature,
        openai_api_key=api_key,
    )

"""Serviço de chat que orquestra ``ConversationStore`` e o chain Langchain."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from app.chat.memory import ConversationStore
from app.chat.schemas import ChatResponse, HistoryMessage, HistoryResponse
from app.core.exceptions import LLMUnavailableError


class ChatService:
    """Orquestra a sessão de chat: gera ``session_id``, invoca o chain e lê o histórico.

    O chain é injetado já montado (``build_chat_chain``) para facilitar testes com LLM fake.
    """

    def __init__(
        self,
        store: ConversationStore,
        chain: Runnable[dict[str, Any], str],
        model_name: str,
    ) -> None:
        self._store = store
        self._chain = chain
        self._model_name = model_name

    async def ask(self, session_id: str | None, message: str) -> ChatResponse:
        """Processa uma mensagem do usuário e devolve a resposta do LLM.

        Quando ``session_id`` é ``None`` ou vazio, gera um novo UUID para a sessão.
        Exceções do chain (LLM indisponível, timeout, etc.) são convertidas em
        ``LLMUnavailableError`` com mensagem descritiva em PT-BR.
        """
        resolved_session_id = session_id if session_id else str(uuid4())

        # Garante que o histórico exista antes do chain invocar o history_factory.
        await self._store.get_or_create(resolved_session_id)

        try:
            answer = await self._chain.ainvoke(
                {"question": message},
                config={"configurable": {"session_id": resolved_session_id}},
            )
        except Exception as exc:
            # Erros do LLM são heterogêneos (rede, auth, timeout, rate limit, etc.);
            # encapsulamos tudo em LLMUnavailableError com mensagem única em PT-BR.
            raise LLMUnavailableError(
                "LLM indisponível: verifique OPENAI_API_KEY e conectividade."
            ) from exc

        return ChatResponse(
            session_id=resolved_session_id,
            answer=answer,
            model=self._model_name,
            created_at=datetime.now(UTC),
        )

    async def get_history(self, session_id: str) -> HistoryResponse:
        """Lê o histórico da sessão e mapeia para o DTO de resposta.

        Levanta ``SessionNotFoundError`` quando a sessão não existe (propagado).
        Timestamps refletem o momento da leitura — limitação conhecida do MVP
        in-memory, já que ``InMemoryChatMessageHistory`` não persiste o instante
        real de cada mensagem.
        """
        history = await self._store.get(session_id)
        now = datetime.now(UTC)

        messages: list[HistoryMessage] = []
        for raw_message in history.messages:
            role: Literal["user", "assistant"]
            if isinstance(raw_message, HumanMessage):
                role = "user"
            elif isinstance(raw_message, AIMessage):
                role = "assistant"
            else:
                # Mensagens de sistema (ou outros tipos) não aparecem no histórico exposto.
                continue

            content = raw_message.content
            if not isinstance(content, str):
                content = str(content)

            messages.append(
                HistoryMessage(role=role, content=content, created_at=now),
            )

        return HistoryResponse(session_id=session_id, messages=messages)

    async def reset(self, session_id: str) -> None:
        """Remove o histórico da sessão; propaga ``SessionNotFoundError`` se inexistente."""
        await self._store.reset(session_id)

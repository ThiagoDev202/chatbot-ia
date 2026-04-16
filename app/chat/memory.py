"""Armazenamento em memória de históricos de conversa por sessão."""

import asyncio

from langchain_core.chat_history import InMemoryChatMessageHistory

from app.core.exceptions import SessionNotFoundError


class ConversationStore:
    """Dicionário async-safe de históricos de conversa indexados por session_id."""

    def __init__(self) -> None:
        self._store: dict[str, InMemoryChatMessageHistory] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str) -> InMemoryChatMessageHistory:
        """Retorna o histórico da sessão, criando-o se ainda não existir."""
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = InMemoryChatMessageHistory()
            return self._store[session_id]

    def get_or_create_sync(self, session_id: str) -> InMemoryChatMessageHistory:
        """Versão síncrona de ``get_or_create`` para uso como factory síncrona.

        O ``RunnableWithMessageHistory`` do Langchain exige uma factory síncrona
        ``(session_id: str) -> BaseChatMessageHistory``. Como o ``asyncio.Lock``
        não pode ser adquirido fora de um contexto async, aqui acessamos o
        dicionário diretamente — o chain não invoca o factory concorrentemente
        para o mesmo ``session_id`` e a criação de um ``dict`` entry é atômica
        o suficiente para o MVP in-memory.
        """
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    async def get(self, session_id: str) -> InMemoryChatMessageHistory:
        """Retorna o histórico da sessão; levanta SessionNotFoundError se inexistente."""
        async with self._lock:
            if session_id not in self._store:
                raise SessionNotFoundError(f"Sessão '{session_id}' não encontrada.")
            return self._store[session_id]

    async def reset(self, session_id: str) -> None:
        """Remove o histórico da sessão; levanta SessionNotFoundError se inexistente."""
        async with self._lock:
            if session_id not in self._store:
                raise SessionNotFoundError(f"Sessão '{session_id}' não encontrada.")
            del self._store[session_id]

    async def exists(self, session_id: str) -> bool:
        """Verifica se a sessão existe no armazenamento."""
        async with self._lock:
            return session_id in self._store

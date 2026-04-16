"""Testes unitários para ConversationStore."""

import asyncio

import pytest

from app.chat.memory import ConversationStore
from app.core.exceptions import SessionNotFoundError


@pytest.fixture
def store() -> ConversationStore:
    """Instância isolada de ConversationStore para cada teste."""
    return ConversationStore()


async def test_get_or_create_creates_new_session(store: ConversationStore) -> None:
    """get_or_create deve criar uma sessão nova com histórico vazio."""
    history = await store.get_or_create("sessao-1")
    assert history is not None
    assert history.messages == []


async def test_get_or_create_is_idempotent(store: ConversationStore) -> None:
    """Duas chamadas com o mesmo session_id devem retornar a mesma instância."""
    first = await store.get_or_create("sessao-1")
    second = await store.get_or_create("sessao-1")
    assert first is second


async def test_get_or_create_preserves_messages_between_calls(
    store: ConversationStore,
) -> None:
    """Mensagens adicionadas via get_or_create devem persistir entre chamadas."""
    history = await store.get_or_create("sessao-1")
    history.add_user_message("Como usar listas em Python?")

    same_history = await store.get_or_create("sessao-1")
    assert len(same_history.messages) == 1
    assert same_history.messages[0].content == "Como usar listas em Python?"


async def test_get_raises_session_not_found_for_missing_id(
    store: ConversationStore,
) -> None:
    """get deve levantar SessionNotFoundError para session_id inexistente."""
    with pytest.raises(SessionNotFoundError):
        await store.get("id-inexistente")


async def test_get_returns_existing_session(store: ConversationStore) -> None:
    """get deve retornar o histórico de uma sessão existente."""
    await store.get_or_create("sessao-1")
    history = await store.get("sessao-1")
    assert history is not None


async def test_reset_removes_existing_session(store: ConversationStore) -> None:
    """Após reset, exists deve retornar False para a sessão removida."""
    await store.get_or_create("sessao-1")
    await store.reset("sessao-1")
    assert await store.exists("sessao-1") is False


async def test_reset_raises_session_not_found_for_missing_id(
    store: ConversationStore,
) -> None:
    """reset deve levantar SessionNotFoundError para session_id inexistente."""
    with pytest.raises(SessionNotFoundError):
        await store.reset("id-inexistente")


async def test_exists_returns_true_for_created_session(
    store: ConversationStore,
) -> None:
    """exists deve retornar True após criação da sessão."""
    await store.get_or_create("sessao-1")
    assert await store.exists("sessao-1") is True


async def test_exists_returns_false_for_missing_id(store: ConversationStore) -> None:
    """exists deve retornar False para session_id inexistente."""
    assert await store.exists("id-inexistente") is False


async def test_concurrent_get_or_create_returns_same_instance(
    store: ConversationStore,
) -> None:
    """Chamadas concorrentes com o mesmo session_id devem retornar a mesma instância."""
    session_id = "sessao-concorrente"
    n = 50
    results = await asyncio.gather(*[store.get_or_create(session_id) for _ in range(n)])
    first = results[0]
    assert all(h is first for h in results), "Todas as tarefas devem retornar a mesma instância"


def test_get_or_create_sync_creates_new_session(store: ConversationStore) -> None:
    """A versão síncrona deve criar uma sessão nova com histórico vazio."""
    history = store.get_or_create_sync("sessao-sync")
    assert history is not None
    assert history.messages == []


def test_get_or_create_sync_is_idempotent_and_shares_with_async(
    store: ConversationStore,
) -> None:
    """A versão síncrona deve devolver a mesma instância da versão async."""
    sync_history = store.get_or_create_sync("sessao-compartilhada")
    sync_history.add_user_message("mensagem via sync")

    # Verifica que a versão async enxerga a mesma estrutura.
    async def _check_async() -> None:
        async_history = await store.get_or_create("sessao-compartilhada")
        assert async_history is sync_history
        assert len(async_history.messages) == 1

    asyncio.run(_check_async())

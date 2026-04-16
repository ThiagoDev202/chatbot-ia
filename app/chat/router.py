"""Endpoints HTTP do chatbot sob o prefixo ``/api/v1/chat``."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.chat.chain import build_chat_chain
from app.chat.memory import ConversationStore
from app.chat.schemas import ChatRequest, ChatResponse, HistoryResponse
from app.chat.service import ChatService
from app.core.config import get_settings

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# Singleton module-level lazy: o ``ChatService`` concreto é construído apenas na
# primeira chamada de ``get_chat_service``. Isso evita instanciar ``ChatOpenAI``
# no import (o que falha em ambientes sem ``OPENAI_API_KEY``, como os testes).
# A API sobe normalmente mesmo sem chave; o 503 só aparece quando alguém faz
# POST em /api/v1/chat sem ter a chave configurada — comportamento exigido
# pelo PRD (seção 4, "503 sem chave OpenAI") e critério de aceite #9.
_service_instance: ChatService | None = None


def _build_chat_service() -> ChatService:
    """Monta o grafo store + chain + service usando as settings em runtime."""
    settings = get_settings()
    store = ConversationStore()
    chain = build_chat_chain(settings, history_factory=store.get_or_create_sync)
    return ChatService(store=store, chain=chain, model_name=settings.openai_model)


def get_chat_service() -> ChatService:
    """Dependência FastAPI: devolve a instância única (lazy) de ``ChatService``."""
    global _service_instance
    if _service_instance is None:
        _service_instance = _build_chat_service()
    return _service_instance


# Alias do ``Depends`` usando ``Annotated`` — padrão recomendado pelo FastAPI e
# aceito pelo ruff (evita o warning B008 de chamada em default de parâmetro).
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Envia uma mensagem ao chatbot.",
)
async def post_chat(request: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    """Recebe ``session_id`` (opcional) e ``message``; devolve a resposta do LLM."""
    return await service.ask(request.session_id, request.message)


@router.get(
    "/{session_id}",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retorna o histórico de uma sessão.",
)
async def get_chat_history(session_id: str, service: ChatServiceDep) -> HistoryResponse:
    """Devolve o histórico completo da sessão em ordem cronológica."""
    return await service.get_history(session_id)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Reseta o histórico de uma sessão.",
)
async def delete_chat_session(session_id: str, service: ChatServiceDep) -> Response:
    """Remove o histórico da sessão; responde 204 sem body."""
    await service.reset(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

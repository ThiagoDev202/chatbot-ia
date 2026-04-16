"""Endpoints HTTP do chatbot sob o prefixo ``/api/v1/chat``."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, status
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from app.chat.chain import build_chat_chain
from app.chat.memory import ConversationStore
from app.chat.schemas import ChatRequest, ChatResponse, HistoryResponse
from app.chat.service import ChatService
from app.core.config import Settings, get_settings
from app.core.exceptions import LLMUnavailableError

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# Singletons module-level. O store existe independente da ``OPENAI_API_KEY`` e
# é reutilizado entre requests (essencial para multi-turn). O chain é cacheado
# apenas após construção bem-sucedida — se a chave estiver ausente, cada POST
# tenta reconstruir (permitindo que o usuário ajuste ``.env`` sem reiniciar).
_store: ConversationStore | None = None
_chain: Runnable[dict[str, Any], str] | None = None


class _NoOpChain(Runnable[dict[str, Any], str]):
    """Chain de fallback usado quando o LLM real não pôde ser construído.

    Qualquer invocação levanta ``LLMUnavailableError``, que o ``ChatService``
    traduz novamente (preservando tipo) e o handler central converte em 503.
    GET e DELETE nunca tocam neste objeto, então continuam funcionando sem
    chave configurada.
    """

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        raise LLMUnavailableError("LLM indisponível: verifique OPENAI_API_KEY e conectividade.")

    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        raise LLMUnavailableError("LLM indisponível: verifique OPENAI_API_KEY e conectividade.")


def _get_store() -> ConversationStore:
    """Devolve o ``ConversationStore`` singleton, criando-o sob demanda."""
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store


def _get_chain(store: ConversationStore, settings: Settings) -> Runnable[dict[str, Any], str]:
    """Devolve o chain real se for possível construí-lo; caso contrário, ``_NoOpChain``.

    Sucesso é cacheado; falha não é — isso permite retry automático quando o
    usuário preencher ``OPENAI_API_KEY`` no ``.env`` sem reiniciar o servidor.
    """
    global _chain
    if _chain is not None:
        return _chain
    # Qualquer falha de construção (auth, rede, validação pydantic) vira _NoOpChain,
    # que levanta LLMUnavailableError na primeira invocação — o handler traduz em 503.
    try:
        _chain = build_chat_chain(settings, history_factory=store.get_or_create_sync)
    except Exception:
        return _NoOpChain()
    return _chain


def get_chat_service() -> ChatService:
    """Dependência FastAPI: monta o ``ChatService`` com o store singleton.

    A construção é barata — apenas agrega referências. O store e o chain reais
    são singletons reutilizados entre requests, garantindo multi-turn correto.
    """
    settings = get_settings()
    store = _get_store()
    chain = _get_chain(store, settings)
    return ChatService(store=store, chain=chain, model_name=settings.openai_model)


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

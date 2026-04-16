"""Entrypoint da aplicacao FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.chat.router import router as chat_router
from app.core.exceptions import (
    ChatbotError,
    LLMUnavailableError,
    SessionNotFoundError,
)
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ciclo de vida da aplicacao: configura logging na inicializacao."""
    setup_logging()
    yield


app = FastAPI(
    title="Chatbot IA",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.include_router(chat_router)


@app.exception_handler(SessionNotFoundError)
async def _handle_session_not_found(
    request: Request,
    exc: SessionNotFoundError,
) -> JSONResponse:
    """Converte ``SessionNotFoundError`` em 404 com detalhe em PT-BR.

    Prefere o ``session_id`` vindo do path param; se indisponível, cai para a
    mensagem da própria exceção ou uma mensagem genérica.
    """
    session_id = request.path_params.get("session_id")
    if session_id:
        detail = f"Sessão '{session_id}' não encontrada."
    else:
        detail = str(exc) if str(exc) else "Sessão não encontrada."
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": detail},
    )


@app.exception_handler(LLMUnavailableError)
async def _handle_llm_unavailable(
    request: Request,
    exc: LLMUnavailableError,
) -> JSONResponse:
    """Converte ``LLMUnavailableError`` em 503 com detalhe em PT-BR."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "LLM indisponível: verifique OPENAI_API_KEY e conectividade."},
    )


@app.exception_handler(ChatbotError)
async def _handle_chatbot_error(
    request: Request,
    exc: ChatbotError,
) -> JSONResponse:
    """Catch-all para erros de domínio não especializados: 500."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erro interno do chatbot."},
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    """Retorna status de saude da API."""
    return {"status": "ok"}

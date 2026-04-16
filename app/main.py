"""Entrypoint da aplicacao FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

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


@app.get("/health")
async def health() -> dict[str, Any]:
    """Retorna status de saude da API."""
    return {"status": "ok"}

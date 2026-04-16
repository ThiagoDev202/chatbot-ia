"""DTOs Pydantic para requests e responses do endpoint de chat."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    """Payload de entrada para ``POST /api/v1/chat``."""

    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str | None = Field(
        default=None,
        description=(
            "Identificador da sessão. Quando omitido, o servidor gera um novo UUID e o "
            "retorna na resposta."
        ),
    )
    message: str = Field(
        min_length=1,
        max_length=4000,
        description="Pergunta ou mensagem do usuário (1 a 4000 caracteres).",
    )

    @field_validator("message", mode="before")
    @classmethod
    def _message_must_not_be_blank(cls, value: object) -> object:
        """Garante que ``message`` não seja vazia ou composta apenas por espaços em branco."""
        if isinstance(value, str) and not value.strip():
            raise ValueError("O campo 'message' não pode estar vazio ou conter apenas espaços.")
        return value


class ChatResponse(BaseModel):
    """Resposta de ``POST /api/v1/chat``."""

    session_id: str = Field(
        description="Identificador da sessão (gerado pelo servidor quando ausente no request).",
    )
    answer: str = Field(
        description="Resposta gerada pelo LLM para a mensagem do usuário.",
    )
    model: str = Field(
        description="Nome do modelo LLM que produziu a resposta (ex.: 'gpt-4o-mini').",
    )
    created_at: datetime = Field(
        description="Timestamp UTC (ISO 8601) do momento em que a resposta foi gerada.",
    )


class HistoryMessage(BaseModel):
    """Mensagem individual no histórico de uma sessão."""

    role: Literal["user", "assistant"] = Field(
        description="Autor da mensagem: 'user' para o usuário e 'assistant' para o tutor.",
    )
    content: str = Field(
        description="Conteúdo textual da mensagem.",
    )
    created_at: datetime = Field(
        description=(
            "Timestamp UTC da mensagem. No MVP in-memory, reflete o momento da leitura "
            "do histórico, não o momento real em que a mensagem foi enviada."
        ),
    )


class HistoryResponse(BaseModel):
    """Resposta de ``GET /api/v1/chat/{session_id}`` com o histórico completo."""

    session_id: str = Field(
        description="Identificador da sessão consultada.",
    )
    messages: list[HistoryMessage] = Field(
        description="Lista de mensagens da sessão em ordem cronológica.",
    )

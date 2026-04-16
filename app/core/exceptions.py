"""Excecoes de dominio do chatbot."""


class ChatbotError(Exception):
    """Erro base do dominio do chatbot."""


class SessionNotFoundError(ChatbotError):
    """Sessao inexistente no armazenamento em memoria."""


class LLMUnavailableError(ChatbotError):
    """O LLM esta indisponivel (sem chave ou falha externa)."""

# Stage 1: builder — instala dependencias de producao com uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copia apenas os arquivos de dependencia primeiro (camada cacheada)
COPY pyproject.toml uv.lock ./

# Instala dependencias de producao no diretorio do projeto
RUN uv sync --frozen --no-dev

# Copia o codigo da aplicacao
COPY app ./app

# Stage 2: runtime — imagem minima sem uv nem ferramentas de build
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Cria usuario nao-root para segurança
RUN groupadd --system appgroup && useradd --system --gid appgroup appuser

# Copia apenas o venv e o codigo do stage anterior
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app

# Garante que o venv esteja no PATH
ENV PATH="/app/.venv/bin:$PATH"

# Usa usuario sem privilegios
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

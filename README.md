# Chatbot IA — API de Tutoria em Python com LLM

API REST assíncrona que expõe um chatbot especializado em responder perguntas sobre programação Python. O chatbot utiliza um modelo de linguagem (LLM) da OpenAI via Langchain, mantém contexto de conversa entre turnos por `session_id`, e é observável via Langsmith.

Construído com **FastAPI + Langchain + OpenAI + Langsmith**, empacotado em Docker e coberto por testes unitários e de integração (46 testes, cobertura global de 99%). O projeto serve como referência educacional de como integrar FastAPI com Langchain e memória de conversação multi-turn seguindo boas práticas de qualidade de software.

---

## Funcionalidades

- `POST /api/v1/chat` — envia mensagem ao chatbot com sessão opcional e contexto multi-turn preservado
- `GET /api/v1/chat/{session_id}` — retorna histórico completo de mensagens da sessão
- `DELETE /api/v1/chat/{session_id}` — reseta o histórico de uma sessão existente
- `GET /health` — liveness check da API
- Tracing via Langsmith (opcional, ativado por variável de ambiente)
- Documentação interativa via Swagger em `/docs`

---

## Arquitetura

```
Client -> FastAPI Router -> ChatService -> Langchain Chain -> ChatOpenAI
                                |
                                +-> ConversationStore (dict in-memory)
```

- **Router** (`app/chat/router.py`) — definição dos endpoints HTTP, validação via Pydantic, injeção do serviço
- **Service** (`app/chat/service.py`) — orquestração: recuperar ou criar sessão, invocar chain, persistir turnos
- **Chain** (`app/chat/chain.py`) — pipeline Langchain com prompt do tutor Python, retorna um `Runnable`
- **ConversationStore** (`app/chat/memory.py`) — dicionário async-safe (`asyncio.Lock`) de históricos por sessão
- **Core** (`app/core/`) — configuração via `pydantic-settings`, exceções de domínio, logging estruturado

---

## Pré-requisitos

- Python 3.12 ou superior
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — gerenciador de pacotes e ambientes virtuais
- Docker e Docker Compose (opcional, para rodar em container)
- Conta na OpenAI com chave de API válida: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

---

## Como obter e configurar a chave da OpenAI

1. Acesse [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. Clique em **"Create new secret key"** e copie o valor gerado (começa com `sk-`).
3. Copie o arquivo de exemplo para `.env`:
   ```bash
   cp .env.example .env
   ```
4. Abra o arquivo `.env` e preencha a variável:
   ```dotenv
   OPENAI_API_KEY=sk-...
   ```
5. Opcional: preencha `LANGCHAIN_API_KEY` se quiser tracing no Langsmith ([https://smith.langchain.com](https://smith.langchain.com)).
6. **Nunca** commite o arquivo `.env` — ele já está listado no `.gitignore`.

> **Aviso:** Sem `OPENAI_API_KEY` válida, o endpoint `/api/v1/chat` retornará erro 503. O restante da API (incluindo `/health` e `/docs`) continua funcional.

---

## Variáveis de ambiente

| Nome | Obrigatória | Default | Descrição |
|------|-------------|---------|-----------|
| `OPENAI_API_KEY` | sim (runtime) | — | Chave da API OpenAI. Sem ela, `/chat` retorna 503 |
| `OPENAI_MODEL` | não | `gpt-4o-mini` | Identificador do modelo LLM a ser usado |
| `OPENAI_TEMPERATURE` | não | `0.3` | Temperatura do modelo (0.0 = determinístico, 1.0 = criativo) |
| `LANGCHAIN_TRACING_V2` | não | `true` | Liga tracing para Langsmith |
| `LANGCHAIN_API_KEY` | não | — | Chave do Langsmith. Sem ela, tracing fica inerte |
| `LANGCHAIN_PROJECT` | não | `chatbot-ia` | Nome do projeto no painel do Langsmith |
| `APP_ENV` | não | `dev` | Ambiente de execução (`dev`, `prod`) |

---

## Como rodar — ambiente local

```bash
# 1. Clonar o repositório
git clone https://github.com/ThiagoDev202/chatbot-ia.git
cd chatbot-ia

# 2. Instalar dependências
uv sync

# 3. Configurar variáveis de ambiente (ver seção "Como obter e configurar a chave da OpenAI")
cp .env.example .env
# Edite .env e preencha OPENAI_API_KEY

# 4. Subir a API
uv run uvicorn app.main:app --reload --port 8000
```

Após subir, acesse `http://localhost:8000/docs` para a documentação interativa Swagger.

---

## Como rodar — Docker

```bash
# 1. Preparar variáveis de ambiente (mesma instrução acima)
cp .env.example .env
# Edite .env e preencha OPENAI_API_KEY

# 2. Build e inicialização
docker compose up --build
```

A API ficará disponível na porta 8000. Swagger: `http://localhost:8000/docs`.

---

## Exemplos de uso

### Exemplo 1 — Pergunta simples (nova sessão)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como criar uma lista em Python?"}'
```

Resposta (o texto gerado varia conforme o modelo; o formato é fixo):

```json
{
  "session_id": "3f2b1a4c-1234-4abc-8def-000000000001",
  "answer": "Em Python, você pode criar uma lista usando colchetes `[]` ou a função `list()`. Exemplos:\n\n```python\nfrutas = [\"maca\", \"banana\", \"laranja\"]\nnumeros = list(range(5))\nquadrados = [x**2 for x in range(1, 6)]\n```\n\nVocê também pode usar `.append()`, `.insert()` e slicing para manipular a lista...",
  "model": "gpt-4o-mini",
  "created_at": "2026-04-16T14:30:01.123456Z"
}
```

Guarde o `session_id` retornado para continuar a conversa no mesmo contexto.

### Exemplo 2 — Follow-up multi-turn

Usando o `session_id` do exemplo anterior, o chatbot mantém o contexto da conversa:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "3f2b1a4c-1234-4abc-8def-000000000001", "message": "E como adicionar elementos?"}'
```

```json
{
  "session_id": "3f2b1a4c-1234-4abc-8def-000000000001",
  "answer": "Para adicionar elementos à lista que criamos, você pode usar `.append(x)` para adicionar no final, `.insert(i, x)` para uma posição específica, ou `.extend(iteravel)` para múltiplos elementos de uma vez:\n\n```python\nfrutas = [\"maca\", \"banana\"]\nfrutas.append(\"laranja\")\nfrutas.insert(1, \"uva\")\nfrutas.extend([\"melao\", \"abacaxi\"])\n```",
  "model": "gpt-4o-mini",
  "created_at": "2026-04-16T14:30:15.789012Z"
}
```

### Exemplo 3 — Consultar histórico da sessão

```bash
curl http://localhost:8000/api/v1/chat/3f2b1a4c-1234-4abc-8def-000000000001
```

```json
{
  "session_id": "3f2b1a4c-1234-4abc-8def-000000000001",
  "messages": [
    {
      "role": "user",
      "content": "Como criar uma lista em Python?",
      "created_at": "2026-04-16T14:30:00.000000Z"
    },
    {
      "role": "assistant",
      "content": "Em Python, você pode criar uma lista usando colchetes `[]` ou a função `list()`...",
      "created_at": "2026-04-16T14:30:01.123456Z"
    },
    {
      "role": "user",
      "content": "E como adicionar elementos?",
      "created_at": "2026-04-16T14:30:14.000000Z"
    },
    {
      "role": "assistant",
      "content": "Para adicionar elementos à lista que criamos, você pode usar `.append(x)`...",
      "created_at": "2026-04-16T14:30:15.789012Z"
    }
  ]
}
```

### Exemplo 4 — Resetar sessão

```bash
curl -X DELETE http://localhost:8000/api/v1/chat/3f2b1a4c-1234-4abc-8def-000000000001
```

Retorna `204 No Content` (sem body). O `session_id` pode ser reutilizado, criando um novo histórico vazio.

---

## Desenvolvimento e testes

```bash
# Lint
uv run ruff check .

# Formatação
uv run ruff format .

# Tipagem estática
uv run mypy app

# Testes com cobertura
uv run pytest -v --cov=app --cov-report=term-missing
```

Para instalar os hooks de pre-commit (opcional, mas recomendado):

```bash
uv run pre-commit install
```

O projeto conta com **46 testes**, cobertura global de **99%** e cobertura de **100%** em `app/chat/`.

---

## Limitações conhecidas (MVP)

- Histórico de conversa vive em memória no processo — perdido em restart ou reimplantação.
- Sem autenticação ou rate limiting: qualquer cliente que conheça o `session_id` acessa o histórico.
- Sem streaming: a resposta é entregue em JSON único após o LLM concluir a geração.
- Sem RAG: o chatbot responde com o conhecimento interno do modelo, sem consultar documentação externa.
- Single-node: o store in-memory não suporta múltiplas réplicas ou escalonamento horizontal.

---

## Evoluções futuras

As próximas iterações planejadas incluem persistência do histórico em Redis (para sobreviver a restarts e escalar horizontalmente), respostas em streaming via Server-Sent Events, RAG integrado à documentação oficial do Python, autenticação por JWT ou API keys, e suporte a múltiplos provedores de LLM além da OpenAI.

---

## Troubleshooting

- **503 "LLM indisponível"** — verifique se `OPENAI_API_KEY` está preenchida corretamente no `.env` e se o modelo definido em `OPENAI_MODEL` existe na sua conta OpenAI.
- **404 "Sessão '...' não encontrada"** — o `session_id` informado não existe no servidor. Lembre-se de que o histórico é mantido em memória: reiniciar o servidor apaga todas as sessões.
- **Langsmith não registra traces** — `LANGCHAIN_TRACING_V2=true` sozinho não é suficiente; é necessário também uma `LANGCHAIN_API_KEY` válida obtida em [https://smith.langchain.com](https://smith.langchain.com).

---

## Licença

Distribuído sob licença MIT.

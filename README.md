# Dealership Sales Agent

[![CI](https://github.com/maxwellnasci/dealership-sales-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/maxwellnasci/dealership-sales-agent/actions/workflows/ci.yml)

Agente de vendas para **revendas de veículos**, atendendo leads por WhatsApp.
Construído com **LangGraph** como uma máquina de estados de qualificação de
lead — conversas naturais, confiáveis, com custo e qualidade mensuráveis.

> Capacidades: StateGraph com arestas condicionais, checkpointing de memória
> entre mensagens, human-in-the-loop (interrupt), tool calling com retry e
> anti-alucinação com Pydantic.

<p align="center">
  <img src="docs/demo.svg" alt="Demonstração animada: conversa do agente no terminal" width="820">
</p>

<sub>Animação gerada por `scripts/gen_demo_svg.py` a partir de uma conversa real (DeepSeek via OpenRouter).</sub>

---

## Por que ele existe (a dor)

Um assistente de vendas precisa **lembrar** do lead entre mensagens
fragmentadas de WhatsApp, **nunca inventar** preço ou condição, saber **chamar
um humano** na hora certa, e ser **medido** (custo e qualidade). Este repo é uma
implementação enxuta e testada desses quatro pilares.

## Arquitetura em 30 segundos

```
WhatsApp ─► supervisor (router, structured output)
                │
     ┌──────────┼───────────┬──────────────┬─────────────┐
     ▼          ▼           ▼              ▼             ▼
 qualify   search_      schedule_     financing     handoff_human
 _lead     inventory    test_drive    (PRICE)       (interrupt → vendedor)
     └──────────┴───────────┴──────────────┘             │
                     ▼                                    │
             synthesize_reply ──► guardrail anti-alucinação ──► END
```

Detalhes e justificativas em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Pilares (e onde ver no código)

| Pilar | Implementação |
|---|---|
| **StateGraph / nós / arestas condicionais** | `graph/builder.py` |
| **Reducers de estado** | `graph/state.py` — `add_messages` + `merge_lead_profile` (merge incremental) |
| **Checkpointing entre mensagens** | `persistence/checkpointer.py` — memory/sqlite/postgres, `thread_id` = telefone |
| **Human-in-the-loop (interrupt)** | `graph/nodes/handoff_human.py` — `interrupt()` + `Command(resume=...)` |
| **Tool calling com retry** | `tools/inventory.py` (tenacity); `tools/financing.py`, `tools/calendar.py` |
| **Anti-alucinação com Pydantic** | `schemas/*` (camada 1) + `guardrails/validator.py` (camada 2) |
| **Async / Pydantic / pytest** | tudo `async`, perfis/tools tipados, `tests/` |
| **Observabilidade (Langfuse)** | `observability/tracing.py` — custo + score por conversa |
| **Avaliação (LLM-as-judge)** | `llm.py::judge` + `tests/test_llm_judge.py` |

## Como rodar

### 1. Demo em terminal (zero setup, roda offline)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/demo.py
```

Sem `OPENAI_API_KEY`, roda em **fallback determinístico**. Com chave (qualquer
endpoint OpenAI-compatível), as respostas ficam naturais.

### 2. Testes

```bash
pytest           # rápido, sem rede, sem API key — usa LLM scriptado
```

### 3. API (produção-like, com Postgres)

```bash
cp .env.example .env            # ajuste o que quiser
docker compose up --build
curl -X POST localhost:8000/webhook -H 'content-type: application/json' \
     -d '{"phone":"+5541999990001","text":"quero um SUV ate 140 mil"}'
```

Handoff: quando o grafo pausa, o `/webhook` devolve `interrupted: true` + o
contexto; o vendedor responde via `POST /resume`.

## Decisões de projeto

- **Ports & adapters no LLM** (`llm.py`): o grafo depende de uma interface
  `LLMClient`, não da SDK. Produção usa `with_structured_output`/`bind_tools`
  do LangChain; testes usam um cliente scriptado → suíte determinística, grátis
  e sem rede.
- **Anti-alucinação em duas camadas**: estrutural (preços só vêm de tools
  tipadas) + validação do texto final contra os fatos do estado. Pior caso =
  resposta mais seca, nunca um preço mentiroso.
- **Memória = checkpointer por `thread_id`**: trocar SQLite↔Postgres não muda
  uma linha do grafo.

## Licença

MIT.

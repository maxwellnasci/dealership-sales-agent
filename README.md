# Dealership Sales Agent

[![CI](https://github.com/maxwellnasci/dealership-sales-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/maxwellnasci/dealership-sales-agent/actions/workflows/ci.yml)
![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/pytest-46%20passed-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

Agente de vendas para **revendas de veículos**, atendendo leads por WhatsApp.
Construído com **LangGraph** como uma máquina de estados de qualificação de
lead — conversas naturais, confiáveis, com custo e qualidade mensuráveis.

> Capacidades: StateGraph com arestas condicionais, roteamento **System-1 Jev**
> (`typesafe/jev-1.14`), checkpointing de memória entre mensagens,
> human-in-the-loop (`interrupt`), tool calling com retry e anti-alucinação
> com Pydantic.

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
WhatsApp ─► supervisor (System-1 Jev 1.14 / structured output fallback)
                │  (~400-600ms | US$ 0.000018/decisão | p calibrada)
     ┌──────────┼───────────┬──────────────┬─────────────┐
     ▼          ▼           ▼              ▼             ▼
 qualify   search_      schedule_     financing     handoff_human
 _lead     inventory    test_drive    (PRICE)       (interrupt → vendedor)
     └──────────┴───────────┴──────────────┘             │
                     ▼                                    │
             synthesize_reply ──► guardrail anti-alucinação ──► END
```

Detalhes, *unit economics* e justificativas em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Pilares (e onde ver no código)

| Pilar | Implementação |
|---|---|
| **StateGraph / nós / arestas condicionais** | `graph/builder.py` |
| **Roteamento System-1 (Jev 1.14)** | `graph/router_jev.py` — `typesafe/jev-1.14` (`/api/alpha/decisions`) + probabilidade calibrada e fallback automático |
| **Reducers de estado** | `graph/state.py` — `add_messages` + `merge_lead_profile` (merge incremental com detecção de mudança de modelo) |
| **Checkpointing entre mensagens** | `persistence/checkpointer.py` — memory/sqlite/postgres, `thread_id` = telefone |
| **Human-in-the-loop (interrupt)** | `graph/nodes/handoff_human.py` — `interrupt()` + `Command(resume=...)` |
| **Tool calling com retry** | `tools/inventory.py` (tenacity); `tools/financing.py`, `tools/calendar.py` |
| **Anti-alucinação em 3 camadas** | `schemas/*` (estrutural) + `_PERSONA` + `guardrails/validator.py` (preço/ano/km amarrados ao veículo citado; data+hora do slot) |
| **Suíte determinística (146 testes, 96% de cobertura)** | `tests/` — tudo `async`, perfis/tools tipados, zero rede e zero API key |
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

### 2. Testes (146 testes determinísticos)

```bash
pytest           # rápido, sem rede, sem API key — usa LLM scriptado
pytest --cov     # com cobertura (gate de 90% no CI)
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

### 4. Benchmark de Roteamento (System-1 Jev vs. LLM Generativo)

Em vez de gastar uma chamada generativa completa (`/chat/completions`) apenas
para decidir a rota do turno, o adaptador [`JevRouter`](src/dealership_agent/graph/router_jev.py)
utiliza o modelo de decisão **`typesafe/jev-1.14`** via endpoint
`/api/alpha/decisions` do OpenRouter, retornando distribuição de probabilidades
calibrada e fallback transparente:

```bash
python scripts/benchmark_router.py
```

| Métrica por Decisão de Rota | LLM Generativo (`structured_output`) | System-1 Decision (`typesafe/jev-1.14`) |
|---|---|---|
| **Latência média** | ~800 – 1.500 ms | **~400 – 600 ms** |
| **Custo médio por decisão** | ~US$ 0.00015+ | **US$ 0.000018** (~55 decisões por US$ 0.001) |
| **Confiança calibrada (`p`)** | Não nativo | **Sim (`probabilities` + `confidence` 0..1)** |
| **Resiliência sem API key / erro** | Fallback `TemplateLLMClient` | **Degradação graciosa para o roteador clássico** |

## Decisões de projeto

- **Arquitetura Híbrida System-1 / System-2**: decisão de rota ultrarrápida e
  barata via modelo classificatório dedicado (`typesafe/jev-1.14`), cálculos
  financeiros e de estoque 100% determinísticos em Python puro (Tabela PRICE), e
  LLM generativo reservado exclusivamente para extração semântica e humanização
  da resposta final.
- **Ports & adapters no LLM** (`llm.py` + `graph/router_jev.py`): o grafo
  depende de uma interface `LLMClient`, não da SDK. Produção usa `JevRouter` +
  `with_structured_output` do LangChain; testes usam um cliente scriptado →
  suíte de testes determinística, grátis e sem rede.
- **Anti-alucinação em três camadas**: estrutural (preços só vêm de tools
  tipadas) + persona restritiva sobre rascunho factual + validação do texto
  final contra os fatos **de cada entidade citada** (`guardrails/validator.py`). Pior caso =
  resposta mais seca, nunca um preço mentiroso.
- **Memória = checkpointer por `thread_id`**: trocar SQLite↔Postgres não muda
  uma linha do grafo.

## Licença

Distribuído sob a licença **MIT** — veja [`LICENSE`](LICENSE) para o texto completo.

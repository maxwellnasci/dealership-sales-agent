# Arquitetura

## Visão geral

O agente é um **StateGraph** no padrão *supervisor + especialistas*. Cada
mensagem de WhatsApp é um turno: o estado é ressuscitado do checkpoint (por
`thread_id = telefone`), o supervisor decide a intenção, um nó especialista faz
o trabalho (sempre via tool tipada), e o nó de síntese produz a resposta
validada contra alucinação.

```
START
  └─► supervisor ──(RouteDecision: Intent)──┐
                                            │
   ┌───────────────┬───────────────┬────────┴──────┬───────────────┐
   ▼               ▼               ▼               ▼               ▼
 qualify_lead   search_inventory schedule_test   financing      handoff_human
   │(tem               │          _drive            │             │ interrupt()
   │ critério?)        │           │                │             ▼
   ├─► search_inventory│           │                │            END (resposta
   └─► synthesize_reply│           │                │             do vendedor)
                       └───────────┴────────────────┘
                                   ▼
                          synthesize_reply
                                   │ (guardrail anti-alucinação)
                                   ▼
                                  END
```

## Estado e reducers (`graph/state.py`)

`GraphState` é um `TypedDict`. Dois campos acumulam via reducer:

- **`messages`** (`add_messages`): histórico da conversa.
- **`lead_profile`** (`merge_lead_profile`): perfil do lead. O reducer faz
  *merge incremental* — nunca apaga um dado já coletado porque a mensagem nova
  não o repetiu. Numa conversa de WhatsApp (uma informação por vez), isso é o
  que mantém o agente coerente ao longo do funil.

O resto (`retrieved_vehicles`, `financing_quote`, `test_drive_slot`,
`final_reply`, `guardrail_flags`) é recomputado por turno; o checkpointer
persiste o estado inteiro, então valores úteis (ex: veículos recuperados)
ficam disponíveis no turno seguinte até serem sobrescritos.

## Memória / checkpointing (`persistence/checkpointer.py`)

Factory com três backends atrás da mesma interface `BaseCheckpointSaver`:
`memory` (testes), `sqlite` (demo auto-contida), `postgres` (produção). O
`thread_id` amarra cada conversa à sua memória. Trocar de backend não toca o
grafo — só a config.

## Human-in-the-loop (`graph/nodes/handoff_human.py`)

Usa o `interrupt()` **nativo** do LangGraph: o turno é suspenso, o checkpoint
guarda o ponto exato, e o chamador recebe um payload com o contexto do lead
para o operador humano. O vendedor responde e o chamador retoma com
`Command(resume=<texto>)` — o valor vira o retorno de `interrupt()` e a fala do
vendedor segue ao lead. Diferente de um "confirme no próximo turno": aqui o
mesmo turno fica pausado e retomável.

## Tool calling com retry (`tools/`)

- `inventory.py`: busca no catálogo, **com retry** (`tenacity`) só para falhas
  transitórias de acesso (`InventoryUnavailable`) — resultado vazio **não** é
  erro e não dispara retry. Essa distinção (infra vs. negócio) é deliberada.
- `financing.py`: simulação PRICE, 100% determinística e pura.
- `calendar.py`: agenda de test-drive. Sem retry de propósito: reservar é
  escrita com efeito colateral, não se re-tenta cegamente.

## Anti-alucinação em três camadas

1. **Estrutural** (`schemas/`): preços, parcelas, veículos e horários só
   existem como instâncias Pydantic produzidas pelas tools. O LLM não gera
   número; apenas verbaliza o que já está tipado.
2. **Persona restritiva sobre rascunho factual** (`graph/nodes/synthesize_reply.py`):
   `build_draft` monta um texto determinístico a partir do estado e `_PERSONA`
   instrui o modelo a apenas humanizá-lo sem inventar valores ou datas.
3. **Validação determinística** (`guardrails/validator.py`): antes de enviar,
   confere que todo valor monetário (incluindo sufixos como `"55 mil"`) e data
   no texto final correspondem a fatos do estado (ignorando anos de modelo como
   `2022`). Se o LLM introduziu um valor desconhecido, o nó de síntese cai no
   **rascunho determinístico** (seguro por construção). Pior caso: resposta mais
   seca, nunca um preço mentiroso.

## Roteamento Híbrido System-1 / System-2 (`llm.py` + `graph/router_jev.py`)

O grafo separa **decisão rápida de intenção (System 1)** de **extração e
geração empática (System 2)** atrás da porta `LLMClient`:

- **System 1 — Roteador de Decisão (`JevRouter` / `typesafe/jev-1.14`)**:
  quando `OPENROUTER_API_KEY` está configurada, `OpenAILLMClient.decide_route`
  delega a classificação das 6 intenções (`qualify`, `search_inventory`,
  `financing`, `schedule_test_drive`, `handoff_human`, `small_talk`) ao
  endpoint `/api/alpha/decisions` do OpenRouter. Devolve a intenção com
  **probabilidade calibrada (`p ∈ [0, 1]`)**, latência de **~400–600 ms** e
  custo de **~US$ 0,000018 por decisão**. Em qualquer falha de rede ou ausência
  de chave, degrada de forma transparente para o roteador clássico.
- **System 2 — Extração e Síntese (`OpenAILLMClient`)**: usa
  `with_structured_output` (extração de `LeadProfile` enriquecida com
  `_lead_summary`) e `ainvoke` (reescrita empática do rascunho factual). Sem
  credencial, `TemplateLLMClient` mantém a demo offline; nos 46 testes,
  injeta-se um cliente scriptado para determinismo total.

## Unit Economics (FinOps em Escala)

Como o roteamento usa um classificador System-1 dedicado e os cálculos
financeiros/estoque rodam em Python puro, o custo operacional por lead atendido
no WhatsApp é mínimo e previsível:

| Etapa de uma conversa típica (5 turnos) | Chamadas | Custo unitário médio | Subtotal |
|---|---|---|---|
| **Roteamento System-1 (`typesafe/jev-1.14`)** | 5 decisões | ~US$ 0.000018 | ~US$ 0.00009 |
| **Extração de perfil + Síntese (`gpt-4o-mini` / DeepSeek)** | 2 extrações + 4 sínteses | ~US$ 0.00020 | ~US$ 0.00120 |
| **Tools (`inventory`, `financing` PRICE, `calendar`) + Guardrail** | Determinístico (CPU) | US$ 0.00000 | US$ 0.00000 |
| **Total por Lead Qualificado (5 turnos)** | — | — | **< US$ 0.0015 (< R$ 0,01)** |

## Observabilidade (`observability/tracing.py`)

Callback do Langfuse plugado no `ainvoke` mede custo/latência por conversa;
`score_trace` anexa a nota do LLM-as-judge a cada trace. Sem chaves, vira no-op
— nunca quebra a execução.

## Extensibilidade

Adicionar um especialista = 1 nó novo + 1 valor em `Intent` + 1 entrada em
`INTENT_TO_NODE` / `JEV_CRITERIA`. O resto do grafo não muda.

## Caminhos de evolução

- **Confidence gating no supervisor**: usar `JevRouteDecision.probability` (ex:
  se `p < 0.60`, rotear para `qualify` pedindo esclarecimento em vez de acionar
  uma tool com baixa certeza).
- Confirmação explícita antes de reservar test-drive (segundo HITL, estilo
  interrupt de aprovação).
- Retrieval de descrições longas dos veículos (RAG) com citação de fonte.
- Avaliação contínua: suíte de LLM-as-judge rodando no CI, scores no Langfuse.
- Multi-revenda: `thread_id` composto (revenda + telefone) e catálogo por
  tenant.

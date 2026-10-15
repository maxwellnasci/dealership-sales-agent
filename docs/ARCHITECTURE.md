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

## Anti-alucinação em duas camadas

1. **Estrutural** (`schemas/`): preços, parcelas, veículos e horários só
   existem como instâncias Pydantic produzidas pelas tools. O LLM não gera
   número; apenas verbaliza o que já está tipado.
2. **Validação** (`guardrails/validator.py`): antes de enviar, confere que todo
   valor monetário no texto final corresponde a um fato do estado. Se o LLM
   introduziu um valor desconhecido, o nó de síntese cai no **rascunho
   determinístico** (seguro por construção). Pior caso: resposta mais seca,
   nunca um preço mentiroso.

## LLM como porta (`llm.py`)

Ports & adapters: o grafo depende de `LLMClient` (abstrato). O adaptador real
(`OpenAILLMClient`) usa `with_structured_output` (roteamento/extração tipados)
e `ainvoke` (composição). Sem credencial, um `TemplateLLMClient` offline mantém
a demo viva. Nos testes, injeta-se um cliente scriptado → determinismo total.

## Observabilidade (`observability/tracing.py`)

Callback do Langfuse plugado no `ainvoke` mede custo/latência por conversa;
`score_trace` anexa a nota do LLM-as-judge a cada trace. Sem chaves, vira no-op
— nunca quebra a execução.

## Extensibilidade

Adicionar um especialista = 1 nó novo + 1 valor em `Intent` + 1 entrada em
`INTENT_TO_NODE`. O resto do grafo não muda.

## Caminhos de evolução

- Confirmação explícita antes de reservar test-drive (segundo HITL, estilo
  interrupt de aprovação).
- Retrieval de descrições longas dos veículos (RAG) com citação de fonte.
- Avaliação contínua: suíte de LLM-as-judge rodando no CI, scores no Langfuse.
- Multi-revenda: `thread_id` composto (revenda + telefone) e catálogo por
  tenant.

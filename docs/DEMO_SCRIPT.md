# Roteiro da demo (vídeo)

Duração alvo: **3–4 min**. Objetivo: mostrar os 4 pilares em ação, numa
conversa que um CTO reconhece como real.

## Preparação

```bash
source .venv/bin/activate
pytest -q                 # mostra a suíte verde primeiro (confiança)
python scripts/demo.py    # roda as duas conversas
```

> Dica: grave uma vez em modo fallback (offline) e, se tiver chave, outra com
> `OPENAI_API_KEY` setada, para mostrar o salto de naturalidade.

## Cena 1 — Descoberta + estoque + financiamento + agendamento (lead 1)

Fala do apresentador enquanto roda:

1. **"Oi, boa tarde!"** → o agente cumprimenta e pergunta o que o lead procura.
   *Destacar:* roteamento por intenção (supervisor).
2. **"Procuro um SUV até 140 mil, pra família"** → extrai perfil (carroceria +
   orçamento) e já lista opções do estoque com preço real.
   *Destacar:* reducer de perfil + tool de estoque tipada.
3. **"Gostei do primeiro. Como fica o financiamento?"** → simula a parcela.
   *Destacar:* **memória** (ele lembra do SUV sem repetir) + número vindo da
   tool PRICE, não do LLM.
4. **"Dá pra agendar um test-drive?"** → confirma data/hora.
   *Destacar:* fluxo completo de qualificação → fechamento.

## Cena 2 — Human-in-the-loop (lead 2)

1. **"Queria um Corolla"** → agente apresenta.
2. **"Consigo um desconto à vista? Quero falar com um vendedor"** → o grafo
   **pausa** (`interrupt`), mostra o contexto do lead para o operador.
   *Destacar:* `interrupt()` nativo + checkpoint guardando o ponto exato.
3. O "vendedor Carlos" responde → o grafo **retoma** (`Command(resume=...)`) e a
   fala do humano vai ao lead.
   *Destacar:* handoff real, não um chatbot que finge escalar.

## Cena 3 — Confiabilidade (fechamento técnico)

Abrir `tests/test_anti_hallucination.py` e rodar:

```bash
pytest tests/test_anti_hallucination.py -q
```

Narrar: "Mesmo que o LLM tente oferecer um preço que não existe, o guardrail
detecta e substitui pela resposta factual. O agente nunca mente um preço."

Fechar com `tests/test_llm_judge.py`: "E a qualidade é medida — LLM-as-judge dá
nota a cada resposta, pronto para rodar no CI e alimentar o Langfuse."

## Frases de efeito para a entrevista

- "Memória é um checkpointer por telefone; trocar SQLite por Postgres não muda
  uma linha do grafo."
- "Anti-alucinação é estrutural: preço não passa pelo LLM, ele só verbaliza o
  que a tool tipada devolveu."
- "O human-in-the-loop usa o `interrupt` nativo — o turno fica suspenso e
  retomável, não é um fork manual."
- "Os testes não gastam token: o LLM é uma porta, nos testes injeto um
  adaptador scriptado."

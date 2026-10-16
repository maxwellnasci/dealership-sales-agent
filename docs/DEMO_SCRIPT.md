# Walkthrough da demo

Percurso curto (~3–4 min) que exercita os quatro pilares do agente numa
conversa de ponta a ponta.

## Preparação

```bash
source .venv/bin/activate
pytest -q                 # suíte verde
python scripts/demo.py    # roda as duas conversas
```

> A demo roda offline (fallback determinístico). Com `OPENAI_API_KEY` definida,
> as respostas passam a ser geradas por um LLM real.

## Cenário 1 — Descoberta, estoque, financiamento e agendamento

1. **"Oi, boa tarde!"** → o agente cumprimenta e pergunta o que o lead procura.
   Exercita: roteamento por intenção (supervisor).
2. **"Procuro um SUV até 140 mil, pra família"** → extrai o perfil (carroceria
   e orçamento) e lista opções do estoque com preço real.
   Exercita: reducer de perfil + tool de estoque tipada.
3. **"Gostei do primeiro. Como fica o financiamento?"** → simula a parcela.
   Exercita: memória entre mensagens (o carro é lembrado sem repetição) e
   valores vindos da tool de financiamento, não do LLM.
4. **"Dá pra agendar um test-drive?"** → confirma um horário específico.
   Exercita: fluxo de qualificação até o agendamento.

## Cenário 2 — Human-in-the-loop

1. **"Queria um Corolla"** → o agente apresenta a opção.
2. **"Consigo um desconto à vista? Quero falar com um vendedor"** → o grafo
   pausa (`interrupt`) e devolve o contexto do lead ao operador.
   Exercita: `interrupt()` nativo + checkpoint preservando o ponto de parada.
3. A resposta do vendedor humano retoma o grafo (`Command(resume=...)`) e
   segue ao lead.
   Exercita: transbordo real, com estado preservado.

## Cenário 3 — Confiabilidade

```bash
pytest tests/test_anti_hallucination.py -q
```

Demonstra que, mesmo quando o LLM tenta citar um preço inexistente, o guardrail
detecta e substitui pela resposta factual — o agente não afirma um valor que
não esteja no estado.

```bash
pytest tests/test_llm_judge.py -q
```

Demonstra o esqueleto de avaliação contínua (LLM-as-judge), pronto para rodar
no CI e alimentar o Langfuse com um score por conversa.

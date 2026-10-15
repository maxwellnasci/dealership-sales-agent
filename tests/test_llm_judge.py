"""LLM-as-judge: avaliação offline da qualidade das respostas (diferencial).

Roda uma conversa real pelo grafo e pede a um "juiz" (aqui determinístico, em
produção um LLM) uma nota de 0 a 1 contra uma rubrica. É o esqueleto de uma
suíte de avaliação contínua ligável ao Langfuse (score_trace).
"""

from __future__ import annotations

from dealership_agent.llm import TemplateLLMClient

RUBRICA = (
    "O agente (1) cumprimenta, (2) apresenta opções com preço real, "
    "(3) avança com uma pergunta (financiamento/test-drive)."
)


async def test_juiz_pontua_resposta_boa(make_graph, new_thread_config, run_turn):
    graph = make_graph()
    config = new_thread_config()
    state = await run_turn(graph, "Quero um SUV até 140 mil", config)
    reply = state["final_reply"]

    judge = TemplateLLMClient()
    score = await judge.judge(RUBRICA, reply)
    assert 0.0 <= score <= 1.0
    assert score >= 0.7  # resposta boa: tem preço, pergunta de avanço


async def test_juiz_penaliza_resposta_pobre():
    judge = TemplateLLMClient()
    score = await judge.judge(RUBRICA, "ok")
    assert score < 0.7

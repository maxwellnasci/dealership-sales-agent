"""Human-in-the-loop: o grafo PAUSA no handoff e RETOMA com o resume.

Prova o uso nativo de `interrupt()` + `Command(resume=...)`: o turno fica
suspenso (checkpoint), devolve o contexto ao operador, e a mensagem do
vendedor humano retomada vira a resposta ao lead.
"""

from __future__ import annotations

from langgraph.types import Command

from dealership_agent.schemas.routing import Intent
from tests.conftest import ScriptedLLMClient


async def test_handoff_interrompe_e_retoma(make_graph, new_thread_config):
    graph = make_graph(ScriptedLLMClient(forced_intent=Intent.HANDOFF))
    config = new_thread_config("+5541999990002")

    # 1) Mensagem do lead que dispara o handoff -> grafo PAUSA.
    from dealership_agent.graph.state import initial_state

    result = await graph.ainvoke(
        initial_state("Quero falar com um vendedor", config["configurable"]["thread_id"]), config
    )
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["reason"] == "handoff_requested"
    assert "lead" in payload
    # Ainda NÃO há resposta final: estamos suspensos esperando o humano.
    assert not result.get("final_reply")

    # 2) O vendedor humano responde -> RETOMA do ponto exato.
    rep_text = "Oi! Aqui é o Carlos. Consigo R$ 3.000 de desconto à vista, fechamos?"
    resumed = await graph.ainvoke(Command(resume=rep_text), config)
    assert resumed["final_reply"] == rep_text


async def test_resume_vazio_cai_em_mensagem_padrao(make_graph, new_thread_config):
    graph = make_graph(ScriptedLLMClient(forced_intent=Intent.HANDOFF))
    config = new_thread_config()
    from dealership_agent.graph.state import initial_state

    tid = config["configurable"]["thread_id"]
    await graph.ainvoke(initial_state("me passa pro atendente", tid), config)
    resumed = await graph.ainvoke(Command(resume="  "), config)
    assert "vendedor" in resumed["final_reply"].lower()

"""Montagem do StateGraph do agente de vendas.

Topologia (padrão supervisor + especialistas):

    START -> supervisor -[route]-> qualify_lead ----[tem critério?]--> search_inventory
                                 |                                  \\-> synthesize_reply
                                 |-> search_inventory --------------------> synthesize_reply
                                 |-> schedule_test_drive -----------------> synthesize_reply
                                 |-> financing ----------------------------> synthesize_reply
                                 |-> synthesize_reply (small_talk)
                                 \\-> handoff_human --(interrupt)--> END
    synthesize_reply -> END

O `LLMClient` é injetado (ports & adapters): produção passa `OpenAILLMClient`,
testes passam um cliente scriptado. Adicionar um especialista = um nó novo +
uma `Intent` + uma entrada no mapa de roteamento, sem tocar no resto.
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from dealership_agent.graph.nodes.financing import financing_node
from dealership_agent.graph.nodes.handoff_human import handoff_human_node
from dealership_agent.graph.nodes.qualify_lead import qualify_lead_node
from dealership_agent.graph.nodes.schedule_test_drive import schedule_test_drive_node
from dealership_agent.graph.nodes.search_inventory import search_inventory_node
from dealership_agent.graph.nodes.supervisor import supervisor_node
from dealership_agent.graph.nodes.synthesize_reply import synthesize_reply_node
from dealership_agent.graph.state import GraphState
from dealership_agent.llm import LLMClient
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.routing import Intent

# Intent (saída do supervisor) -> nó de destino.
INTENT_TO_NODE: dict[str, str] = {
    Intent.QUALIFY.value: "qualify_lead",
    Intent.SEARCH_INVENTORY.value: "search_inventory",
    Intent.SCHEDULE_TEST_DRIVE.value: "schedule_test_drive",
    Intent.FINANCING.value: "financing",
    Intent.HANDOFF.value: "handoff_human",
    Intent.SMALL_TALK.value: "synthesize_reply",
}


def route_after_supervisor(state: GraphState) -> str:
    """Lê `state['route']` e devolve o nó de destino (fallback: qualify_lead)."""
    return INTENT_TO_NODE.get(state.get("route") or "", "qualify_lead")


def route_after_qualify(state: GraphState) -> str:
    """Após extrair o perfil: se já há critério de busca, mostra opções;
    senão, volta a perguntar (via síntese)."""
    lead: LeadProfile = state.get("lead_profile") or LeadProfile()
    if lead.desired_make or lead.desired_model or lead.body_type or lead.budget_max is not None:
        return "search_inventory"
    return "synthesize_reply"


def build_graph(checkpointer: BaseCheckpointSaver, llm: LLMClient):
    """Compila o grafo com o checkpointer e o cliente de LLM injetados."""
    graph = StateGraph(GraphState)

    graph.add_node("supervisor", partial(supervisor_node, llm=llm))
    graph.add_node("qualify_lead", partial(qualify_lead_node, llm=llm))
    graph.add_node("search_inventory", partial(search_inventory_node, llm=llm))
    graph.add_node("schedule_test_drive", partial(schedule_test_drive_node, llm=llm))
    graph.add_node("financing", partial(financing_node, llm=llm))
    graph.add_node("handoff_human", partial(handoff_human_node, llm=llm))
    graph.add_node("synthesize_reply", partial(synthesize_reply_node, llm=llm))

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "qualify_lead": "qualify_lead",
            "search_inventory": "search_inventory",
            "schedule_test_drive": "schedule_test_drive",
            "financing": "financing",
            "handoff_human": "handoff_human",
            "synthesize_reply": "synthesize_reply",
        },
    )
    graph.add_conditional_edges(
        "qualify_lead",
        route_after_qualify,
        {"search_inventory": "search_inventory", "synthesize_reply": "synthesize_reply"},
    )

    graph.add_edge("search_inventory", "synthesize_reply")
    graph.add_edge("schedule_test_drive", "synthesize_reply")
    graph.add_edge("financing", "synthesize_reply")
    graph.add_edge("synthesize_reply", END)
    graph.add_edge("handoff_human", END)  # handoff já produz final_reply (pós-resume)

    return graph.compile(checkpointer=checkpointer)

"""Nó supervisor (router).

Pede ao LLM uma `RouteDecision` tipada e grava a intenção em `state["route"]`.
Não executa trabalho de negócio — só decide o próximo nó. A aresta condicional
`route_after_supervisor` (builder.py) lê `state["route"]`.
"""

from __future__ import annotations

from dealership_agent.graph.state import GraphState
from dealership_agent.llm import LLMClient
from dealership_agent.schemas.lead import LeadProfile


async def supervisor_node(state: GraphState, *, llm: LLMClient) -> dict:
    messages = state.get("messages", [])
    lead = state.get("lead_profile") or LeadProfile()
    decision = await llm.decide_route(messages, lead)
    return {"route": decision.intent.value}

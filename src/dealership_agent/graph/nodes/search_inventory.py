"""Nó de busca de estoque.

Monta a `InventoryQuery` a partir do perfil acumulado do lead e chama a tool
(com retry). O resultado vai para `retrieved_vehicles` — a única fonte de que
o nó de síntese pode falar sobre carros/preços.
"""

from __future__ import annotations

from dealership_agent.config import settings
from dealership_agent.graph.nodes.qualify_lead import _last_user_text
from dealership_agent.graph.state import GraphState
from dealership_agent.llm import LLMClient
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.tools import InventoryQuery
from dealership_agent.tools.inventory import search_inventory


def query_from_lead(lead: LeadProfile) -> InventoryQuery:
    return InventoryQuery(
        make=lead.desired_make,
        model=lead.desired_model,
        body_type=lead.body_type,
        max_price=lead.budget_max,
    )


async def search_inventory_node(state: GraphState, *, llm: LLMClient) -> dict:
    """Busca o estoque aplicando os critérios do perfil.

    Extrai critérios da mensagem ATUAL antes de consultar: o lead pode pedir
    "me mostra SUVs até 120 mil" numa tacada só (sem ter passado por uma etapa
    de descoberta separada), e a busca precisa refletir isso. O delta extraído
    também volta ao estado (reducer `merge_lead_profile`).
    """
    current = state.get("lead_profile") or LeadProfile()
    extracted = await llm.extract_lead(_last_user_text(state.get("messages", [])), current)
    effective = current.merged_with(extracted)
    vehicles = search_inventory(query_from_lead(effective), limit=settings.max_vehicle_suggestions)
    return {"retrieved_vehicles": vehicles, "lead_profile": extracted}

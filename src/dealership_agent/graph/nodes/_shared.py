"""Helpers compartilhados entre nós de especialista."""

from __future__ import annotations

from dealership_agent.config import settings
from dealership_agent.graph.state import GraphState
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.tools import Vehicle
from dealership_agent.tools.inventory import search_inventory


def ensure_vehicles(state: GraphState) -> list[Vehicle]:
    """Veículos disponíveis para este turno.

    Usa `retrieved_vehicles` se a busca já rodou (nesta conversa, persistido
    pelo checkpoint); senão faz uma busca sob demanda a partir do perfil —
    assim o lead pode pedir financiamento direto sem ter pedido a lista antes.
    """
    existing = state.get("retrieved_vehicles") or []
    if existing:
        return existing
    lead = state.get("lead_profile") or LeadProfile()
    from dealership_agent.graph.nodes.search_inventory import query_from_lead

    return search_inventory(query_from_lead(lead), limit=settings.max_vehicle_suggestions)

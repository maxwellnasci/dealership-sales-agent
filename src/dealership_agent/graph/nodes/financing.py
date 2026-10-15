"""Nó de simulação de financiamento.

Escolhe um veículo (o mais aderente ao perfil/orçamento) e chama a tool
determinística. Grava `financing_quote`, que o nó de síntese verbaliza e o
guardrail usa como fonte da verdade dos valores.
"""

from __future__ import annotations

from dealership_agent.graph.nodes._shared import ensure_vehicles
from dealership_agent.graph.state import GraphState
from dealership_agent.llm import LLMClient
from dealership_agent.tools.financing import simulate_financing


async def financing_node(state: GraphState, *, llm: LLMClient) -> dict:
    vehicles = ensure_vehicles(state)
    if not vehicles:
        return {"financing_quote": None}
    # Entrada padrão da revenda (regra de negócio na tool); o veículo é o mais
    # aderente ao perfil, já ordenado por preço na busca.
    quote = simulate_financing(vehicles[0])
    return {"financing_quote": quote, "retrieved_vehicles": vehicles}

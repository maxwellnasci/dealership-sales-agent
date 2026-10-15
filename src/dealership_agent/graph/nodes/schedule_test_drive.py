"""Nó de agendamento de test-drive.

Pega o primeiro horário disponível para o veículo de interesse e o confirma.
Grava `test_drive_slot` para o nó de síntese anunciar data/hora reais.
"""

from __future__ import annotations

from dealership_agent.graph.nodes._shared import ensure_vehicles
from dealership_agent.graph.state import GraphState
from dealership_agent.llm import LLMClient
from dealership_agent.tools.calendar import available_slots, book_slot


async def schedule_test_drive_node(state: GraphState, *, llm: LLMClient) -> dict:
    vehicles = ensure_vehicles(state)
    if not vehicles:
        return {"test_drive_slot": None}
    slots = available_slots(vehicles[0])
    if not slots:
        return {"test_drive_slot": None}
    confirmed = book_slot(slots[0])
    return {"test_drive_slot": confirmed, "retrieved_vehicles": vehicles}

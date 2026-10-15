"""Cada intenção roteia para o nó/comportamento correto."""

from __future__ import annotations

import pytest

from dealership_agent.graph.builder import (
    INTENT_TO_NODE,
    route_after_qualify,
    route_after_supervisor,
)
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.routing import Intent


@pytest.mark.parametrize("intent", list(Intent))
def test_toda_intent_tem_no_de_destino(intent: Intent):
    assert intent.value in INTENT_TO_NODE


def test_route_after_supervisor_mapeia_intent():
    assert route_after_supervisor({"route": Intent.FINANCING.value}) == "financing"
    assert route_after_supervisor({"route": Intent.HANDOFF.value}) == "handoff_human"
    assert route_after_supervisor({"route": Intent.SMALL_TALK.value}) == "synthesize_reply"


def test_route_after_supervisor_fallback_seguro():
    assert route_after_supervisor({"route": None}) == "qualify_lead"
    assert route_after_supervisor({}) == "qualify_lead"


def test_route_after_qualify_vai_buscar_quando_ha_criterio():
    state = {"lead_profile": LeadProfile(body_type="suv")}
    assert route_after_qualify(state) == "search_inventory"


def test_route_after_qualify_pergunta_quando_sem_criterio():
    state = {"lead_profile": LeadProfile(name="Ana")}  # nome não é critério de busca
    assert route_after_qualify(state) == "synthesize_reply"

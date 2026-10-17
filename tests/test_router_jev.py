"""Roteador Jev: mapeamento tipado, parse do payload e fallback clássico."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from dealership_agent.graph.builder import INTENT_TO_NODE
from dealership_agent.graph.router_jev import (
    JEV_LABEL_TO_INTENT,
    JevRouteDecision,
    JevRouter,
    parse_jev_payload,
)
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.routing import Intent


def test_labels_jev_mapeiam_para_intents_do_grafo():
    assert JEV_LABEL_TO_INTENT["search_inventory"] is Intent.SEARCH_INVENTORY
    assert JEV_LABEL_TO_INTENT["financing"] is Intent.FINANCING
    assert JEV_LABEL_TO_INTENT["schedule_test_drive"] is Intent.SCHEDULE_TEST_DRIVE
    # handoff_human é o nó; a Intent é "handoff" e precisa ter destino no grafo.
    assert JEV_LABEL_TO_INTENT["handoff_human"] is Intent.HANDOFF
    for intent in JEV_LABEL_TO_INTENT.values():
        assert intent.value in INTENT_TO_NODE


def test_parse_payload_decisions_com_probabilidade_e_uso():
    payload = {
        "model": "typesafe/jev-1.14-20261014",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "financing",
                "probabilities": {"financing": 0.9, "search_inventory": 0.1},
                "confidence": 0.9,
            }
        },
        "usage": {"input_tokens": 374, "output_tokens": 55, "cost": 0.000015708},
    }
    label, prob, in_tok, out_tok, cost = parse_jev_payload(payload)
    assert label == "financing"
    assert prob == pytest.approx(0.90)
    assert (in_tok, out_tok) == (374, 55)
    assert cost == pytest.approx(0.000015708)


def test_parse_payload_sem_probabilities_usa_confidence():
    payload = {"answers": {"intent": {"choice": "handoff_human", "confidence": 0.87}}}
    label, prob, in_tok, out_tok, cost = parse_jev_payload(payload)
    assert label == "handoff_human"
    assert prob == pytest.approx(0.87)
    assert (in_tok, out_tok, cost) == (0, 0, 0.0)


@pytest.mark.asyncio
async def test_sem_chave_openrouter_cai_no_fallback_classico():
    router = JevRouter(api_key="")  # sem chave: nunca chama a rede
    assert router.configured is False
    decision = await router.route_message("Quero agendar um test-drive no sábado")
    assert decision.source == "fallback-classico"
    assert decision.intent is Intent.SCHEDULE_TEST_DRIVE
    assert 0.0 <= decision.probability <= 1.0
    assert decision.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_decide_route_compativel_com_porta_do_supervisor():
    router = JevRouter(api_key="")
    decision = await router.decide_route(
        [HumanMessage("quero falar com um vendedor")], LeadProfile()
    )
    assert decision.intent is Intent.HANDOFF


@pytest.mark.asyncio
async def test_erro_do_jev_cai_no_fallback(monkeypatch):
    router = JevRouter(api_key="chave-falsa")

    async def _boom(messages):  # simula falha de rede/parse do OpenRouter
        raise RuntimeError("rede indisponível")

    monkeypatch.setattr(router, "_classify", _boom)
    decision = await router.route_message("Consigo financiar com 20 mil de entrada?")
    assert decision.source == "fallback-classico"
    assert decision.intent is Intent.FINANCING


@pytest.mark.asyncio
async def test_sucesso_do_jev_retorna_probabilidade(monkeypatch):
    router = JevRouter(api_key="chave-falsa")

    async def _ok(messages):
        return JevRouteDecision(
            intent=Intent.SEARCH_INVENTORY, probability=0.93, label="search_inventory"
        )

    monkeypatch.setattr(router, "_classify", _ok)
    decision = await router.route_message("Vocês têm SUV até 150 mil?")
    assert decision.source == "jev"
    assert decision.intent is Intent.SEARCH_INVENTORY
    assert decision.probability == pytest.approx(0.93)

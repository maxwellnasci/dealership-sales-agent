"""Roteador Jev: mapeamento tipado, parse do payload e fallback clássico."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from dealership_agent.graph.builder import INTENT_TO_NODE
from dealership_agent.graph.router_jev import (
    JEV_CRITERIA,
    JEV_LABEL_TO_INTENT,
    JEV_LABELS,
    JevRouteDecision,
    JevRouter,
    parse_jev_payload,
)
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.routing import Intent


def test_labels_jev_mapeiam_para_intents_do_grafo():
    assert set(JEV_LABELS) == {
        "qualify",
        "search_inventory",
        "financing",
        "schedule_test_drive",
        "handoff_human",
        "small_talk",
    }
    assert set(JEV_CRITERIA.keys()) == set(JEV_LABELS)
    assert JEV_LABEL_TO_INTENT["qualify"] is Intent.QUALIFY
    assert JEV_LABEL_TO_INTENT["search_inventory"] is Intent.SEARCH_INVENTORY
    assert JEV_LABEL_TO_INTENT["financing"] is Intent.FINANCING
    assert JEV_LABEL_TO_INTENT["schedule_test_drive"] is Intent.SCHEDULE_TEST_DRIVE
    # handoff_human é o nó; a Intent é "handoff" e precisa ter destino no grafo.
    assert JEV_LABEL_TO_INTENT["handoff_human"] is Intent.HANDOFF
    assert JEV_LABEL_TO_INTENT["small_talk"] is Intent.SMALL_TALK
    for label in JEV_LABELS:
        assert JEV_LABEL_TO_INTENT[label].value in INTENT_TO_NODE
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


# --- Caminho HTTP real (`_classify`) com httpx.MockTransport -----------------
# O request passa pelo httpx de verdade (serialização, headers, status), só o
# transporte é falso: nenhuma conexão sai da máquina.


@pytest.fixture
def openrouter(monkeypatch):
    """Intercepta o AsyncClient do httpx; `respond` define a resposta do mock."""
    import httpx

    calls: dict = {"requests": [], "respond": None}
    real_client = httpx.AsyncClient

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["requests"].append(request)
        return calls["respond"](request)

    def _client(**kwargs):
        calls["timeout"] = kwargs.get("timeout")
        return real_client(transport=httpx.MockTransport(_handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client)
    return calls


def _decisions_payload(choice: str, probability: float = 0.97) -> dict:
    return {
        "answers": {
            "intent": {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: probability},
                "confidence": probability,
            }
        },
        "usage": {"input_tokens": 374, "output_tokens": 55, "cost": 0.000015},
    }


@pytest.mark.asyncio
async def test_http_monta_request_de_decisao_e_parseia_resposta(openrouter):
    import json

    import httpx

    from dealership_agent.graph.router_jev import OPENROUTER_DECISIONS_URL

    openrouter["respond"] = lambda req: httpx.Response(200, json=_decisions_payload("financing"))
    router = JevRouter(api_key="or-key", model="typesafe/jev-test", timeout_s=3.0)

    decision = await router.route_message("Quanto fica a parcela?")

    assert decision.source == "jev"
    assert decision.intent is Intent.FINANCING
    assert decision.probability == pytest.approx(0.97)
    assert (decision.prompt_tokens, decision.completion_tokens) == (374, 55)
    assert decision.total_tokens == 429
    assert decision.cost_usd == pytest.approx(0.000015)
    assert decision.latency_ms >= 0.0

    (req,) = openrouter["requests"]
    assert req.method == "POST"
    assert str(req.url) == OPENROUTER_DECISIONS_URL
    assert req.headers["Authorization"] == "Bearer or-key"
    assert req.headers["X-Title"] == "dealership-sales-agent"
    body = json.loads(req.content)
    assert body["model"] == "typesafe/jev-test"
    assert body["state"] == "Lead: Quanto fica a parcela?"
    question = body["questions"]["intent"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == list(JEV_LABELS)
    # Timeout explícito chega ao cliente HTTP.
    assert openrouter["timeout"] == 3.0


@pytest.mark.asyncio
async def test_http_historico_vira_state_multilinha(openrouter):
    import json

    import httpx

    openrouter["respond"] = lambda req: httpx.Response(200, json=_decisions_payload("qualify"))
    router = JevRouter(api_key="or-key")
    await router.decide_route(
        [HumanMessage("Oi"), HumanMessage("Quero um SUV")], LeadProfile()
    )
    body = json.loads(openrouter["requests"][0].content)
    assert body["state"] == "Lead: Oi\nLead: Quero um SUV"


@pytest.mark.asyncio
async def test_http_normaliza_rotulo_handoff(openrouter):
    import httpx

    payload = _decisions_payload("Handoff-Human")
    openrouter["respond"] = lambda req: httpx.Response(200, json=payload)
    decision = await JevRouter(api_key="or-key").route_message("Quero um vendedor")
    assert decision.source == "jev"
    assert decision.intent is Intent.HANDOFF
    assert decision.label == "handoff_human"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "http_500", "http_401", "timeout", "conexao",
        "rotulo_desconhecido", "sem_answers", "nao_json",
    ],
)
async def test_http_falhas_caem_no_fallback_classico(openrouter, scenario):
    import httpx

    def _respond(req: httpx.Request) -> httpx.Response:
        if scenario == "http_500":
            return httpx.Response(500, json={"error": "upstream"})
        if scenario == "http_401":
            return httpx.Response(401, json={"error": "invalid key"})
        if scenario == "timeout":
            raise httpx.ReadTimeout("lento demais", request=req)
        if scenario == "conexao":
            raise httpx.ConnectError("dns", request=req)
        if scenario == "rotulo_desconhecido":
            return httpx.Response(200, json=_decisions_payload("comprar_moto"))
        if scenario == "sem_answers":
            return httpx.Response(200, json={"usage": {}})
        return httpx.Response(200, text="<html>bad gateway</html>")

    openrouter["respond"] = _respond
    decision = await JevRouter(api_key="or-key").route_message("Consigo financiar?")

    assert openrouter["requests"], "deveria ter tentado o Jev antes do fallback"
    assert decision.source == "fallback-classico"
    assert decision.intent is Intent.FINANCING


def test_get_router_usa_modelo_do_settings(monkeypatch):
    from dealership_agent.config import settings
    from dealership_agent.graph.router_jev import get_router

    monkeypatch.setattr(settings, "jev_model", "typesafe/jev-9")
    assert get_router()._model == "typesafe/jev-9"

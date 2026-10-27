"""Integração do mxos_ldm com o StateGraph: nó predict_lead_scoring,
persistência no checkpoint (allowlist) e borda da API.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from dealership_agent import main
from dealership_agent.config import settings
from dealership_agent.graph.builder import build_graph
from dealership_agent.graph.nodes.predict_lead_scoring import predict_lead_scoring_node
from dealership_agent.graph.state import initial_state
from dealership_agent.llm import TemplateLLMClient
from dealership_agent.persistence.checkpointer import checkpoint_serde, checkpointer_context
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.tools.financing import DEFAULT_DOWN_PAYMENT_PCT
from mxos_ldm.data_engine import generate_dataset
from mxos_ldm.model import LDMPredictor
from mxos_ldm.sales_bridge import SellerCard, build_seller_card
from mxos_ldm.schemas import PredictionOutput, VehicleAppraisalInput

PHONE = "+5541988887777"
MSG = "Quero um SUV até 140 mil"
TRADE_IN = {
    "ano_modelo": 2019,
    "km_rodados": 78_000,
    "valor_fipe": 62_500.0,
    "percentual_fipe_pedido": 0.95,
    "descricao": "VW Gol 1.6 2019",
}


@pytest.fixture(scope="module")
def predictor():
    p = LDMPredictor(backend="sklearn")
    p.fit(generate_dataset(3_000, seed=3))
    return p


def _graph(predictor=None, saver=None):
    return build_graph(
        saver or InMemorySaver(serde=checkpoint_serde()), TemplateLLMClient(), predictor
    )


def _config(phone: str = PHONE) -> dict:
    return {"configurable": {"thread_id": phone}}


async def test_turno_com_troca_gera_previsao_e_card(predictor):
    graph = _graph(predictor)
    vehicle = VehicleAppraisalInput(**TRADE_IN)
    result = await graph.ainvoke(initial_state(MSG, PHONE, vehicle), _config())

    assert result["route"] == "qualify"
    prediction = result["ldm_prediction"]
    card = result["seller_briefing_card"]
    assert isinstance(prediction, PredictionOutput)
    assert isinstance(card, SellerCard)
    assert not card.bloqueado
    assert card.lead_id == PHONE
    assert f"Entrada: {DEFAULT_DOWN_PAYMENT_PCT * 100:.0f}% do valor" in card.texto
    # Números do LDM são do vendedor: nunca vazam para a resposta do lead.
    assert "Avaliação máxima" not in result["final_reply"]
    assert "fechamento" not in result["final_reply"]


async def test_turno_sem_troca_nao_preve(predictor):
    result = await _graph(predictor).ainvoke(initial_state(MSG, PHONE), _config())
    assert result.get("ldm_prediction") is None
    assert result.get("seller_briefing_card") is None


async def test_sem_predictor_no_op():
    vehicle = VehicleAppraisalInput(**TRADE_IN)
    result = await _graph().ainvoke(initial_state(MSG, PHONE, vehicle), _config())
    assert result["ldm_vehicle_input"] == vehicle
    assert result.get("ldm_prediction") is None
    assert result.get("final_reply")  # fluxo segue igual


async def test_checkpoint_preserva_estado_ldm_entre_turnos(predictor, caplog):
    graph = _graph(predictor)
    vehicle = VehicleAppraisalInput(**TRADE_IN)
    with caplog.at_level(logging.WARNING):
        await graph.ainvoke(initial_state(MSG, PHONE, vehicle), _config())
        # Turno 2 sem trade_in: o carro da troca persistido é reaproveitado.
        await graph.ainvoke(initial_state("prefiro um sedan", PHONE), _config())
        snapshot = await graph.aget_state(_config())

    values = snapshot.values
    assert values["ldm_vehicle_input"] == vehicle
    assert isinstance(values["ldm_prediction"], PredictionOutput)
    assert isinstance(values["seller_briefing_card"], SellerCard)
    assert "unregistered type" not in caplog.text


async def test_checkpoint_sqlite_real(predictor, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "checkpoint_backend", "sqlite")
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "ckpt.sqlite"))
    vehicle = VehicleAppraisalInput(**TRADE_IN)

    async with checkpointer_context() as saver:
        first = await _graph(predictor, saver).ainvoke(
            initial_state(MSG, PHONE, vehicle), _config()
        )

    async with checkpointer_context() as saver:
        snapshot = await _graph(predictor, saver).aget_state(_config())

    assert snapshot.values["ldm_vehicle_input"] == vehicle
    assert snapshot.values["ldm_prediction"] == first["ldm_prediction"]
    assert snapshot.values["seller_briefing_card"] == first["seller_briefing_card"]


def test_serde_roundtrip_tipos_ldm(predictor):
    vehicle = VehicleAppraisalInput(**TRADE_IN)
    lead = {"lead_id": PHONE, "valor_entrada_pct": 0.3}
    prediction = predictor.predict_raw(lead, TRADE_IN)
    card = build_seller_card(
        {"ldm_lead": lead, "ldm_vehicle": vehicle, "ldm_prediction": prediction}
    )
    serde = checkpoint_serde()
    for value in (vehicle, prediction, card):
        assert serde.loads_typed(serde.dumps_typed(value)) == value


@pytest.mark.parametrize("fipe", [None, float("nan"), 0.0])
async def test_no_fail_closed_com_fipe_invalida(predictor, fipe):
    state = {
        "lead_profile": LeadProfile(phone=PHONE, name="Ana"),
        "ldm_vehicle_input": {**TRADE_IN, "valor_fipe": fipe},
    }
    out = await predict_lead_scoring_node(state, predictor=predictor)
    assert out["ldm_prediction"] is None
    assert out["seller_briefing_card"].bloqueado
    assert out["seller_briefing_card"].motivo_bloqueio.startswith("FIPE indisponível")


async def test_no_usa_interesse_do_perfil(predictor):
    state = {
        "lead_profile": LeadProfile(desired_make="Jeep", desired_model="Renegade"),
        "ldm_vehicle_input": VehicleAppraisalInput(**TRADE_IN),
    }
    out = await predict_lead_scoring_node(state, predictor=predictor)
    card = out["seller_briefing_card"]
    assert "Interesse: Jeep Renegade" in card.texto
    assert card.lead_id == "desconhecido"


# ---------------------------------------------------------------- API


@pytest.fixture
def ldm_client(monkeypatch, predictor, tmp_path):
    monkeypatch.setattr(settings, "checkpoint_backend", "memory")
    monkeypatch.setattr(settings, "scheduler_db_path", str(tmp_path / "scheduler.sqlite"))
    monkeypatch.setattr(main, "get_llm_client", TemplateLLMClient)
    monkeypatch.setattr(main, "get_langfuse_handler", lambda: None)
    monkeypatch.setattr(main, "build_ldm_predictor", lambda: predictor)
    with TestClient(main.app) as c:
        yield c


def test_webhook_com_trade_in(ldm_client):
    resp = ldm_client.post(
        "/webhook", json={"phone": PHONE, "text": MSG, "trade_in": TRADE_IN}
    )
    assert resp.status_code == 200
    snapshot = main.app.state.graph.get_state(_config())
    assert isinstance(snapshot.values["seller_briefing_card"], SellerCard)


def test_webhook_trade_in_sem_fipe_recusado_na_borda(ldm_client):
    resp = ldm_client.post(
        "/webhook",
        json={"phone": PHONE, "text": MSG, "trade_in": {**TRADE_IN, "valor_fipe": None}},
    )
    assert resp.status_code == 422


def test_build_ldm_predictor(monkeypatch):
    monkeypatch.setattr(settings, "ldm_enabled", False)
    assert main.build_ldm_predictor() is None

    monkeypatch.setattr(settings, "ldm_enabled", True)
    monkeypatch.setattr(settings, "ldm_backend", "sklearn")
    monkeypatch.setattr(main, "generate_dataset", lambda: generate_dataset(1_000))
    predictor = main.build_ldm_predictor()
    assert predictor.backend == "sklearn"
    assert predictor.weights.versao == "mxos_ldm-1.0.0"
    assert predictor.predict_raw({"lead_id": PHONE, "valor_entrada_pct": 0.2}, TRADE_IN)

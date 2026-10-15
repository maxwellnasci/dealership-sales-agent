"""Guardrail anti-alucinação: valores não-verificados são barrados."""

from __future__ import annotations

from dealership_agent.guardrails.validator import validate_reply
from dealership_agent.schemas.tools import FinancingQuote, Vehicle
from tests.conftest import ScriptedLLMClient

_VEHICLE = Vehicle(
    vehicle_id="V001", make="Toyota", model="Corolla", year=2022,
    price=129900.0, mileage_km=38000, body_type="sedan", color="prata",
)


def test_validator_aprova_valor_do_estoque():
    reply = "O Corolla sai por R$ 129.900,00."
    assert validate_reply(reply, [_VEHICLE], None) == []


def test_validator_barra_valor_inventado():
    reply = "Consigo esse Corolla por apenas R$ 55.000!"
    flags = validate_reply(reply, [_VEHICLE], None)
    assert any(f.startswith("unverified_amount") for f in flags)


def test_validator_aprova_parcela_da_simulacao():
    quote = FinancingQuote(
        vehicle_id="V001", vehicle_label="Toyota Corolla 2022", price=129900.0,
        down_payment=25980.0, term_months=48, monthly_rate=0.019,
        monthly_payment=3200.0, total_paid=179580.0,
    )
    reply = "A parcela fica em R$ 3.200,00 em 48x."
    assert validate_reply(reply, [], quote) == []


def test_validator_ignora_ano_e_prazo():
    # 2022 e "48 meses" não são valores monetários de preço.
    reply = "Temos o Corolla 2022, financiável em 48 meses."
    assert validate_reply(reply, [_VEHICLE], None) == []


async def test_no_de_sintese_volta_ao_rascunho_quando_llm_alucina(
    make_graph, new_thread_config, run_turn
):
    # LLM "malicioso" que injeta um preço inexistente na resposta.
    bogus = "Pode levar esse por R$ 55.000, oferta imperdível!"
    graph = make_graph(ScriptedLLMClient(compose_override=bogus))
    config = new_thread_config()

    state = await run_turn(graph, "Me mostra SUVs até 120 mil", config)
    # O guardrail detectou e trocou pelo rascunho factual.
    assert state["guardrail_flags"]
    assert "55.000" not in state["final_reply"]
    # O rascunho só cita preços reais do estoque recuperado.
    assert state["retrieved_vehicles"]

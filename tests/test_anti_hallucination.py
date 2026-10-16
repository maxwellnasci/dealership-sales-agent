"""Guardrail anti-alucinação: valores não-verificados são barrados."""

from __future__ import annotations

from datetime import datetime

from dealership_agent.guardrails.validator import validate_reply
from dealership_agent.schemas.tools import FinancingQuote, TestDriveSlot, Vehicle
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


def test_validator_barra_inteiro_cru_sem_separador():
    # Regressão do branch shadowing: "55000" (sem ponto) tem que ser pego.
    flags = validate_reply("Faço por 55000 à vista.", [_VEHICLE], None)
    assert any(f.startswith("unverified_amount") for f in flags)


def test_validator_barra_valor_em_mil():
    # "55 mil" = 55000, valor inventado -> barrado.
    flags = validate_reply("Consigo uns 55 mil nesse.", [_VEHICLE], None)
    assert any("unverified_amount" in f for f in flags)


def test_validator_aprova_preco_cru_valido():
    # O preço real escrito sem separador não pode gerar falso positivo.
    assert validate_reply("Sai por 129900.", [_VEHICLE], None) == []


def test_validator_nao_confunde_ano_recente_com_preco():
    reply = "Tenho opções 2019, 2022 e 2024 em estoque."
    assert validate_reply(reply, [_VEHICLE], None) == []


_SLOT = TestDriveSlot(
    slot_id="V001-202610190900", vehicle_id="V001", vehicle_label="Toyota Corolla 2022",
    starts_at=datetime(2026, 10, 19, 9, 0), confirmed=True,
)


def test_validator_aprova_a_data_do_slot():
    reply = "Agendei seu test-drive para 19/10 às 09h00. Confirma?"
    assert validate_reply(reply, [], None, _SLOT) == []


def test_validator_barra_data_alternativa_inventada():
    reply = "Tenho 19/10 às 09h00, mas também posso 20/10 às 14h00."
    flags = validate_reply(reply, [], None, _SLOT)
    assert "unverified_date:20/10" in flags
    assert not any(f == "unverified_date:19/10" for f in flags)


def test_validator_ignora_datas_fora_de_contexto_de_agenda():
    # Sem slot, não checamos datas (evita falso positivo em conversa comum).
    reply = "Podemos conversar dia 25/12 se quiser."
    assert validate_reply(reply, [], None, None) == []


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

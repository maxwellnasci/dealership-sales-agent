"""Guardrail amarrado por entidade: regressões da auditoria Red Team.

A versão anterior validava contra uma allowlist GLOBAL de inteiros e removia
"km" antes de checar. Com dois carros no contexto, o LLM podia colar o preço de
um no outro; podia inventar quilometragem, ano e horário do test-drive. Cada
teste abaixo reproduz um desses bypasses e prova que agora é barrado.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from dealership_agent.graph.nodes.synthesize_reply import build_draft
from dealership_agent.guardrails.validator import validate_reply
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.routing import Intent
from dealership_agent.schemas.tools import FinancingQuote, TestDriveSlot, Vehicle
from dealership_agent.tools import inventory
from dealership_agent.tools.calendar import available_slots, book_slot
from dealership_agent.tools.financing import simulate_financing
from dealership_agent.tools.inventory import InventoryUnavailable, load_catalog

COROLLA = Vehicle(
    vehicle_id="V001", make="Toyota", model="Corolla", year=2022,
    price=129900.0, mileage_km=38000, body_type="sedan", color="prata",
)
POLO = Vehicle(
    vehicle_id="V008", make="Volkswagen", model="Polo", year=2023,
    price=89900.0, mileage_km=15000, body_type="hatch", color="branco",
)
COROLLA_CROSS = Vehicle(
    vehicle_id="V002", make="Toyota", model="Corolla Cross", year=2023,
    price=164900.0, mileage_km=21000, body_type="suv", color="branco",
)
HRV = Vehicle(
    vehicle_id="V004", make="Honda", model="HR-V", year=2022,
    price=139900.0, mileage_km=34000, body_type="suv", color="cinza",
)
# 19/10/2026 é uma segunda-feira.
SLOT = TestDriveSlot(
    slot_id="V001-202610190900", vehicle_id="V001", vehicle_label="Toyota Corolla 2022",
    starts_at=datetime(2026, 10, 19, 9, 0), confirmed=True,
)
QUOTE = FinancingQuote(
    vehicle_id="V001", vehicle_label="Toyota Corolla 2022", price=129900.0,
    down_payment=25980.0, term_months=48, monthly_rate=0.019,
    monthly_payment=3200.0, total_paid=179580.0,
)


# --- Preço cruzado -----------------------------------------------------------


def test_barra_preco_de_outro_veiculo_no_mesmo_contexto():
    # Bypass original: 89.900 existe (é do Polo), mas foi atribuído ao Corolla.
    flags = validate_reply("O Corolla sai por R$ 89.900,00.", [COROLLA, POLO], None)
    assert "unverified_amount:R$ 89.900,00" in flags


def test_aprova_cada_preco_no_segmento_do_seu_veiculo():
    reply = "O Corolla sai por R$ 129.900,00 e o Polo por R$ 89.900,00."
    assert validate_reply(reply, [COROLLA, POLO], None) == []


def test_barra_precos_trocados_entre_dois_veiculos():
    reply = "O Corolla sai por R$ 89.900,00 e o Polo por R$ 129.900,00."
    flags = validate_reply(reply, [COROLLA, POLO], None)
    assert "unverified_amount:R$ 89.900,00" in flags
    assert "unverified_amount:R$ 129.900,00" in flags


def test_barra_preco_em_mil_de_outro_veiculo():
    flags = validate_reply("Esse Polo fica em 129,9 mil.", [COROLLA, POLO], None)
    assert any(f.startswith("unverified_amount") for f in flags)


def test_valores_da_simulacao_valem_so_para_o_veiculo_simulado():
    ok = "No Corolla, entrada de R$ 25.980,00 e parcela de R$ 3.200,00."
    assert validate_reply(ok, [COROLLA, POLO], QUOTE) == []
    bad = "No Polo, entrada de R$ 25.980,00 e parcela de R$ 3.200,00."
    flags = validate_reply(bad, [COROLLA, POLO], QUOTE)
    assert "unverified_amount:R$ 25.980,00" in flags
    assert "unverified_amount:R$ 3.200,00" in flags


def test_veiculo_da_simulacao_e_fato_mesmo_fora_da_lista_recuperada():
    reply = "O Corolla sai por R$ 129.900,00, parcela de R$ 3.200,00."
    assert validate_reply(reply, [], QUOTE) == []


def test_barra_preco_citado_antes_do_modelo_na_mesma_frase():
    # Revisão Red Team do próprio fix: "texto antes da menção" não pode virar
    # zona sem dono quando o modelo vem depois, na mesma frase.
    flags = validate_reply("Por R$ 89.900,00 você leva o Corolla.", [COROLLA, POLO], None)
    assert "unverified_amount:R$ 89.900,00" in flags
    assert validate_reply("Por R$ 129.900,00 você leva o Corolla.", [COROLLA, POLO], None) == []
    flags = validate_reply("Com só 8.000 km, o Corolla é uma joia.", [COROLLA], None)
    assert "unverified_mileage:8.000 km" in flags


def test_frase_sem_modelo_herda_o_ultimo_citado():
    flags = validate_reply("O Corolla é lindo. Sai por R$ 89.900,00.", [COROLLA, POLO], None)
    assert "unverified_amount:R$ 89.900,00" in flags
    reply = "Gostei do Polo! Sai por R$ 89.900,00. E o Corolla? R$ 129.900,00."
    assert validate_reply(reply, [COROLLA, POLO], None) == []


def test_texto_sem_dono_ainda_exige_valor_conhecido():
    assert validate_reply("Temos opções de R$ 89.900,00.", [COROLLA, POLO], None) == []
    flags = validate_reply("Temos opções de R$ 70.000,00.", [COROLLA, POLO], None)
    assert "unverified_amount:R$ 70.000,00" in flags


def test_modelo_mais_longo_tem_prioridade_na_mencao():
    # "Corolla Cross" não pode ser lido como "Corolla" + texto solto.
    reply = "O Corolla Cross 2023 sai por R$ 164.900,00; o Corolla 2022 por R$ 129.900,00."
    assert validate_reply(reply, [COROLLA, COROLLA_CROSS], None) == []
    flags = validate_reply("O Corolla Cross sai por R$ 129.900,00.", [COROLLA, COROLLA_CROSS], None)
    assert "unverified_amount:R$ 129.900,00" in flags


@pytest.mark.parametrize("spelling", ["HR-V", "HRV", "hr v", "Hr-v"])
def test_reconhece_variacoes_de_grafia_do_modelo(spelling):
    flags = validate_reply(f"O {spelling} sai por R$ 89.900,00.", [HRV, POLO], None)
    assert "unverified_amount:R$ 89.900,00" in flags


def test_barra_veiculo_do_catalogo_que_nao_e_fato_do_turno():
    flags = validate_reply("Também temos um Civic por R$ 119.900,00.", [COROLLA], None)
    assert "unverified_vehicle:Civic" in flags
    assert "unverified_amount:R$ 119.900,00" in flags


def test_catalogo_indisponivel_nao_derruba_o_guardrail(monkeypatch):
    def _down():
        raise InventoryUnavailable("CRM fora do ar")

    monkeypatch.setattr(inventory, "_load_catalog", _down)
    # Ainda reconhece os veículos do turno e barra o preço cruzado.
    flags = validate_reply("O Corolla sai por R$ 89.900,00.", [COROLLA, POLO], None)
    assert "unverified_amount:R$ 89.900,00" in flags


# --- Quilometragem -------------------------------------------------------------


def test_barra_quilometragem_inventada():
    # Bypass original: "km" era removido antes da checagem.
    flags = validate_reply("Corolla 2022 com apenas 8.000 km.", [COROLLA], None)
    assert "unverified_mileage:8.000 km" in flags


@pytest.mark.parametrize("km", ["38.000 km", "38000 km", "38000km", "38 mil km", "38 mil KM"])
def test_aprova_quilometragem_real_em_varios_formatos(km):
    assert validate_reply(f"O Corolla está com {km}.", [COROLLA], None) == []


def test_barra_quilometragem_de_outro_veiculo():
    flags = validate_reply("O Corolla tem 15.000 km rodados.", [COROLLA, POLO], None)
    assert "unverified_mileage:15.000 km" in flags


def test_quilometragem_nao_e_confundida_com_preco():
    # "38 mil km" não pode virar R$ 38.000 (unverified_amount).
    flags = validate_reply("Corolla com 38 mil km.", [COROLLA], None)
    assert flags == []


def test_barra_quilometragem_sem_dono_desconhecida():
    flags = validate_reply("Todos abaixo de 5.000 km!", [COROLLA, POLO], None)
    assert "unverified_mileage:5.000 km" in flags


# --- Ano ---------------------------------------------------------------------


def test_barra_ano_errado_colado_no_modelo():
    flags = validate_reply("Corolla 2023 por R$ 129.900,00.", [COROLLA, POLO], None)
    assert "unverified_year:2023" in flags


def test_ano_solto_continua_sem_falso_positivo():
    assert validate_reply("Tenho opções 2019, 2022 e 2024.", [COROLLA], None) == []


def test_ano_dentro_de_data_nao_e_ano_de_modelo():
    reply = "Corolla 2022 agendado para 19/10/2026 às 09h00."
    assert validate_reply(reply, [COROLLA], None, SLOT) == []


# --- Slot de test-drive: data + hora + dia + veículo -------------------------


def test_barra_horario_inventado_no_dia_certo():
    # Bypass original: só DD/MM era checado; "16h" passava.
    flags = validate_reply("Agendado para 19/10 às 16h.", [], None, SLOT)
    assert flags == ["unverified_time:16h00"]


@pytest.mark.parametrize(
    "when", ["09h00", "9h", "9h00", "09:00", "9 horas", "às 9", "às 9 da manhã"]
)
def test_aprova_horario_do_slot_em_varios_formatos(when):
    reply = f"Seu test-drive do Corolla fica para 19/10, {when}."
    assert validate_reply(reply, [COROLLA], None, SLOT) == []


@pytest.mark.parametrize(
    ("when", "flag"),
    [
        ("às 16h30", "unverified_time:16h30"),
        ("às 14:00", "unverified_time:14h00"),
        ("às 4 da tarde", "unverified_time:16h00"),
        ("às 10", "unverified_time:10h00"),
    ],
)
def test_barra_horario_diferente_do_slot(when, flag):
    flags = validate_reply(f"Fica para 19/10 {when}.", [], None, SLOT)
    assert flag in flags


def test_barra_dia_da_semana_inconsistente_com_o_slot():
    assert validate_reply("Segunda-feira, 19/10, às 09h00.", [], None, SLOT) == []
    flags = validate_reply("Sexta-feira, 19/10, às 09h00.", [], None, SLOT)
    assert "unverified_weekday:Sexta-feira" in flags


def test_barra_slot_atribuido_a_outro_veiculo():
    flags = validate_reply("O test-drive do Polo fica 19/10 às 09h00.", [COROLLA, POLO], None, SLOT)
    assert "unverified_slot_vehicle:Polo" in flags


def test_slot_citado_antes_do_veiculo_errado_na_mesma_frase():
    flags = validate_reply("Dia 19/10 às 09h00 você testa o Polo.", [COROLLA, POLO], None, SLOT)
    assert "unverified_slot_vehicle:Polo" in flags
    assert validate_reply("Dia 19/10 às 09h00 você testa o Corolla.", [COROLLA], None, SLOT) == []


def test_prazo_em_horas_maior_que_23_nao_e_horario():
    reply = "Confirmo em até 48h: 19/10 às 09h00."
    assert validate_reply(reply, [], None, SLOT) == []


def test_sem_slot_horarios_nao_sao_conferidos():
    assert validate_reply("Atendemos até 18h.", [], None, None) == []


# --- O rascunho factual nunca pode ser barrado (anti falso positivo) ---------


def _draft(route: Intent, **state) -> str:
    return build_draft({"route": route.value, "lead_profile": LeadProfile(name="Ana"), **state})


@pytest.mark.parametrize("vehicle", load_catalog(), ids=lambda v: v.model)
def test_rascunhos_de_todo_o_catalogo_passam_no_guardrail(vehicle):
    catalog = load_catalog()
    others = [v for v in catalog if v.vehicle_id != vehicle.vehicle_id][:2]
    listing = [vehicle, *others]
    quote = simulate_financing(vehicle)
    slot = book_slot(available_slots(vehicle)[0])

    cases = [
        (_draft(Intent.SEARCH_INVENTORY, retrieved_vehicles=listing), listing, None, None),
        (_draft(Intent.FINANCING, financing_quote=quote), listing, quote, None),
        (_draft(Intent.SCHEDULE_TEST_DRIVE, test_drive_slot=slot), listing, None, slot),
    ]
    for draft, vehicles, q, s in cases:
        assert validate_reply(draft, vehicles, q, s) == [], draft

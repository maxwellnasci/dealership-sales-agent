"""Tool de agenda e nó de test-drive: slot real e guardrail de horário no grafo."""

from __future__ import annotations

from datetime import datetime

from dealership_agent.schemas.routing import Intent
from dealership_agent.schemas.tools import Vehicle
from dealership_agent.tools.calendar import available_slots, book_slot
from tests.conftest import ScriptedLLMClient

_VEHICLE = Vehicle(
    vehicle_id="V001", make="Toyota", model="Corolla", year=2022,
    price=129900.0, mileage_km=38000, body_type="sedan", color="prata",
)


def test_slots_cobrem_horario_comercial_de_cada_dia():
    slots = available_slots(_VEHICLE, from_date=datetime(2026, 11, 3, 8, 0), days=3)
    assert len(slots) == 12
    assert {s.starts_at.hour for s in slots} == {9, 11, 14, 16}
    assert {s.starts_at.day for s in slots} == {3, 4, 5}
    assert slots[0].starts_at == datetime(2026, 11, 3, 9, 0)
    assert all(not s.confirmed for s in slots)
    assert all(s.vehicle_id == "V001" and s.vehicle_label == "Toyota Corolla 2022" for s in slots)


def test_slot_ids_sao_unicos_e_rastreaveis():
    slots = available_slots(_VEHICLE)
    assert len({s.slot_id for s in slots}) == len(slots)
    assert slots[0].slot_id == "V001-202610190900"  # data-base padrão determinística


def test_zero_dias_nao_gera_slot():
    assert available_slots(_VEHICLE, days=0) == []


def test_book_slot_confirma_sem_mutar_e_e_idempotente():
    slot = available_slots(_VEHICLE)[0]
    booked = book_slot(slot)
    assert booked.confirmed is True
    assert slot.confirmed is False  # cópia, não mutação
    assert book_slot(booked) == booked


async def test_no_de_agendamento_reserva_e_anuncia_o_slot(make_graph, new_thread_config, run_turn):
    graph = make_graph(ScriptedLLMClient(forced_intent=Intent.SCHEDULE_TEST_DRIVE))
    state = await run_turn(graph, "Quero agendar um test-drive do Corolla", new_thread_config())

    slot = state["test_drive_slot"]
    assert slot is not None and slot.confirmed
    assert f"{slot.starts_at:%d/%m às %Hh%M}" in state["final_reply"]
    assert state["guardrail_flags"] == []


async def test_no_de_agendamento_sem_veiculo_nao_reserva(
    make_graph, new_thread_config, run_turn, monkeypatch
):
    from dealership_agent.graph.nodes import schedule_test_drive

    monkeypatch.setattr(schedule_test_drive, "ensure_vehicles", lambda state: [])
    graph = make_graph(ScriptedLLMClient(forced_intent=Intent.SCHEDULE_TEST_DRIVE))
    state = await run_turn(graph, "Quero agendar", new_thread_config())
    assert state["test_drive_slot"] is None


async def test_no_de_agendamento_sem_horario_livre_nao_reserva(
    make_graph, new_thread_config, run_turn, monkeypatch
):
    from dealership_agent.graph.nodes import schedule_test_drive

    monkeypatch.setattr(schedule_test_drive, "available_slots", lambda vehicle: [])
    graph = make_graph(ScriptedLLMClient(forced_intent=Intent.SCHEDULE_TEST_DRIVE))
    state = await run_turn(graph, "Quero agendar um test-drive do Corolla", new_thread_config())
    assert state["test_drive_slot"] is None


async def test_llm_que_inventa_horario_cai_no_rascunho(make_graph, new_thread_config, run_turn):
    # Regressão ponta-a-ponta: o dia está certo (19/10), a hora não (16h != 09h00).
    bogus = "Fechado! Seu test-drive do Corolla ficou para 19/10 às 16h."
    graph = make_graph(
        ScriptedLLMClient(forced_intent=Intent.SCHEDULE_TEST_DRIVE, compose_override=bogus)
    )
    state = await run_turn(graph, "Quero agendar um test-drive do Corolla", new_thread_config())

    assert "unverified_time:16h00" in state["guardrail_flags"]
    assert "16h" not in state["final_reply"]
    assert "19/10 às 09h00" in state["final_reply"]

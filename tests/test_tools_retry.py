"""Tool calling com retry (tenacity) e disciplina de quando NÃO re-tentar."""

from __future__ import annotations

from dealership_agent.schemas.tools import InventoryQuery, Vehicle
from dealership_agent.tools import inventory
from dealership_agent.tools.financing import simulate_financing


def test_retry_recupera_de_falha_transitoria(monkeypatch):
    """`search_inventory` re-tenta `InventoryUnavailable` e sucede na 3ª vez."""
    calls = {"n": 0}

    def flaky_load():
        calls["n"] += 1
        if calls["n"] < 3:
            raise inventory.InventoryUnavailable("timeout simulado do CRM")
        return [
            Vehicle(
                vehicle_id="V001", make="Toyota", model="Corolla", year=2022,
                price=129900.0, mileage_km=38000, body_type="sedan", color="prata",
            )
        ]

    monkeypatch.setattr(inventory, "_load_catalog", flaky_load)
    result = inventory.search_inventory(InventoryQuery(make="Toyota"))
    assert calls["n"] == 3  # falhou 2x, sucedeu na 3ª
    assert len(result) == 1


def test_estoque_vazio_nao_e_erro_e_nao_retenta(monkeypatch):
    calls = {"n": 0}

    def empty_load():
        calls["n"] += 1
        return []

    monkeypatch.setattr(inventory, "_load_catalog", empty_load)
    result = inventory.search_inventory(InventoryQuery(make="Ferrari"))
    assert result == []
    assert calls["n"] == 1  # resultado vazio é resposta legítima, sem retry


def test_financing_e_deterministico():
    v = Vehicle(
        vehicle_id="V001", make="Toyota", model="Corolla", year=2022,
        price=100000.0, mileage_km=10000, body_type="sedan", color="prata",
    )
    q1 = simulate_financing(v, down_payment=20000.0, term_months=48, monthly_rate=0.019)
    q2 = simulate_financing(v, down_payment=20000.0, term_months=48, monthly_rate=0.019)
    assert q1 == q2
    assert q1.monthly_payment > 0
    # Total pago > preço (há juros).
    assert q1.total_paid > v.price

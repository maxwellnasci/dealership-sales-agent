"""O reducer de perfil acumula sem nunca apagar dado já coletado."""

from __future__ import annotations

from dealership_agent.graph.state import merge_lead_profile
from dealership_agent.schemas.lead import LeadProfile


def test_merge_preserva_dado_antigo_quando_novo_e_none():
    old = LeadProfile(name="Ana", budget_max=140000.0)
    new = LeadProfile(desired_model="Corolla")  # não repete nome nem orçamento
    merged = merge_lead_profile(old, new)
    assert merged.name == "Ana"
    assert merged.budget_max == 140000.0
    assert merged.desired_model == "Corolla"


def test_merge_novo_valor_sobrescreve_antigo():
    old = LeadProfile(budget_max=100000.0)
    new = LeadProfile(budget_max=150000.0)
    assert merge_lead_profile(old, new).budget_max == 150000.0


def test_merge_com_old_none_usa_new():
    new = LeadProfile(name="João")
    assert merge_lead_profile(None, new).name == "João"


def test_merge_nunca_apaga_com_none_explicito():
    old = LeadProfile(name="Ana", desired_make="Toyota")
    merged = merge_lead_profile(old, LeadProfile())  # tudo None
    assert merged.name == "Ana"
    assert merged.desired_make == "Toyota"

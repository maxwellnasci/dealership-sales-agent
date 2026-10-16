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


def test_troca_de_modelo_invalida_marca_antiga():
    # Lead muda de ideia: Corolla (Toyota) -> Civic, sem citar marca nova.
    old = LeadProfile(desired_make="Toyota", desired_model="Corolla")
    merged = merge_lead_profile(old, LeadProfile(desired_model="Civic"))
    assert merged.desired_model == "Civic"
    assert merged.desired_make is None  # Toyota não gruda no Civic


def test_troca_de_modelo_com_nova_marca_usa_a_nova():
    old = LeadProfile(desired_make="Toyota", desired_model="Corolla")
    merged = merge_lead_profile(old, LeadProfile(desired_make="Honda", desired_model="Civic"))
    assert merged.desired_make == "Honda"
    assert merged.desired_model == "Civic"


def test_mesmo_modelo_mantem_marca():
    # Repetir o mesmo modelo não deve zerar a marca já conhecida.
    old = LeadProfile(desired_make="Toyota", desired_model="Corolla")
    merged = merge_lead_profile(old, LeadProfile(desired_model="Corolla"))
    assert merged.desired_make == "Toyota"

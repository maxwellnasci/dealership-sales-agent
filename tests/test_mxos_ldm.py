"""Suíte de testes unitários e validação empírica estrita do motor `mxos_ldm` (Polo Curitiba)."""

from __future__ import annotations

import pytest

from dealership_agent.schemas.tools import Vehicle
from dealership_agent.tools.financing import simulate_financing
from dealership_agent.tools.mxos_ldm import (
    CURITIBA_MODELS,
    MXOSAutomotiveLDM,
    load_ldm_weights,
)


def test_weights_json_carrega_top10_modelos_curitiba():
    weights = load_ldm_weights()
    assert weights["metadata"]["engine"] == "mxos_ldm"
    expected_models = {
        "Onix",
        "HB20",
        "Polo",
        "Kwid",
        "Tracker",
        "Creta",
        "Compass",
        "Hilux",
        "Corolla",
        "Civic",
    }
    assert set(CURITIBA_MODELS.keys()) == expected_models


def test_cenario_1_onix_2023_linha_verde_baixa_km():
    """1. Onix 2023 (Linha Verde) — 24k km em 2 anos | 82% FIPE | Entrada 15% (<20%)."""
    res = MXOSAutomotiveLDM.evaluate_deal(
        model="Onix",
        fipe_value=84_900.0,
        total_km=24_000,
        vehicle_age_years=2.0,
        fipe_paid_ratio_tau=0.82,
        down_payment_pct=0.15,
        hub="linha_verde",
    )
    assert res.annual_km == 12_000.0
    assert res.km_liquidity_factor == 1.008
    assert res.fipe_paid_ratio == 0.82
    assert res.credit_approval_prob == 0.361
    assert res.effective_monthly_rate == 0.0200
    assert res.expected_days_on_lot == 17.9
    assert res.net_expected_margin_pct == 0.130
    assert res.recommendation == "COMPRA_FORTE"


def test_cenario_2_onix_2023_linha_verde_ex_app():
    """2. Onix 2023 (Linha Verde) — 48k km em 2 anos (Ex-App) | 90% FIPE | Entrada 45% (>40%)."""
    res = MXOSAutomotiveLDM.evaluate_deal(
        model="Onix",
        fipe_value=84_900.0,
        total_km=48_000,
        vehicle_age_years=2.0,
        fipe_paid_ratio_tau=0.90,
        down_payment_pct=0.45,
        hub="linha_verde",
    )
    assert res.annual_km == 24_000.0
    assert res.km_liquidity_factor == 0.749
    assert res.fipe_paid_ratio == 0.90
    assert res.credit_approval_prob == 0.726
    assert res.effective_monthly_rate == 0.0182
    assert res.expected_days_on_lot == 58.9
    assert res.net_expected_margin_pct == -0.086
    assert res.recommendation == "RENEGOCIAR_TROCA"


def test_cenario_3_corolla_2022_mario_tourinho():
    """3. Corolla 2022 (Mário Tourinho) — 38k km em 3 anos | 89% FIPE | Entrada 45% (>40%)."""
    res = MXOSAutomotiveLDM.evaluate_deal(
        model="Corolla",
        fipe_value=129_900.0,
        total_km=38_000,
        vehicle_age_years=3.0,
        fipe_paid_ratio_tau=0.89,
        down_payment_pct=0.45,
        hub="mario_tourinho",
    )
    assert round(res.annual_km) == 12_667
    assert res.km_liquidity_factor == 1.006
    assert res.fipe_paid_ratio == 0.89
    assert res.credit_approval_prob == 0.726
    assert res.effective_monthly_rate == 0.0182
    assert res.expected_days_on_lot == 21.1
    assert res.net_expected_margin_pct == 0.078
    assert res.recommendation == "COMPRA_PADRAO"


def test_cenario_4_compass_2022_mario_tourinho():
    """4. Compass 2022 (Mário Tourinho) — 66k km em 3 anos | 88% FIPE | Entrada 15% (<20%)."""
    res = MXOSAutomotiveLDM.evaluate_deal(
        model="Compass",
        fipe_value=149_900.0,
        total_km=66_000,
        vehicle_age_years=3.0,
        fipe_paid_ratio_tau=0.88,
        down_payment_pct=0.15,
        hub="mario_tourinho",
    )
    assert res.annual_km == 22_000.0
    assert res.km_liquidity_factor == 0.715
    assert res.fipe_paid_ratio == 0.88
    assert res.credit_approval_prob == 0.361
    assert res.effective_monthly_rate == 0.0200
    assert res.expected_days_on_lot == 139.8
    assert res.net_expected_margin_pct == -0.164
    assert res.recommendation == "RENEGOCIAR_TROCA"


def test_penalidade_km_acima_15000_monotonica():
    _, phi_12k = MXOSAutomotiveLDM.compute_km_penalty("HB20", 36_000, 3.0)
    _, phi_15k = MXOSAutomotiveLDM.compute_km_penalty("HB20", 45_000, 3.0)
    _, phi_22k = MXOSAutomotiveLDM.compute_km_penalty("HB20", 66_000, 3.0)
    _, phi_32k = MXOSAutomotiveLDM.compute_km_penalty("HB20", 96_000, 3.0)
    assert phi_12k > phi_15k == 1.0 > phi_22k > phi_32k


def test_correlacao_fipe_troca_80_a_92_vs_tempo_patio():
    dias = [
        MXOSAutomotiveLDM.evaluate_deal(
            model="Polo",
            fipe_value=89_900.0,
            total_km=30_000,
            vehicle_age_years=2.0,
            fipe_paid_ratio_tau=tau,
            down_payment_pct=0.30,
            hub="linha_verde",
        ).expected_days_on_lot
        for tau in (0.80, 0.84, 0.88, 0.92)
    ]
    assert dias[0] <= dias[1] < dias[2] < dias[3]


def test_simulate_financing_integra_credit_approval_e_effective_monthly_rate():
    vehicle = Vehicle(
        vehicle_id="V010",
        make="Chevrolet",
        model="Onix",
        year=2023,
        price=84_900.0,
        mileage_km=12_000,
        body_type="hatch",
        color="preto",
    )
    # Entrada < 20% (15%)
    q_sub20 = simulate_financing(vehicle, down_payment=84_900.0 * 0.15, term_months=48)
    # Entrada > 40% (45%)
    q_over40 = simulate_financing(vehicle, down_payment=84_900.0 * 0.45, term_months=48)

    assert q_sub20.credit_approval_prob == 0.361
    assert q_sub20.monthly_rate == 0.0200

    assert q_over40.credit_approval_prob == 0.726
    assert q_over40.monthly_rate == 0.0182

    # Maior entrada -> maior probabilidade de aprovação, menor taxa e menor parcela
    assert q_over40.credit_approval_prob > q_sub20.credit_approval_prob
    assert q_over40.monthly_rate < q_sub20.monthly_rate
    assert q_over40.monthly_payment < q_sub20.monthly_payment


@pytest.mark.parametrize("bad_fipe", [0.0, -10_000.0, float("nan"), float("inf")])
def test_evaluate_deal_rejeita_fipe_invalida(bad_fipe):
    with pytest.raises(ValueError, match="fipe_value"):
        MXOSAutomotiveLDM.evaluate_deal(
            model="Civic",
            fipe_value=bad_fipe,
            total_km=40_000,
            vehicle_age_years=3.0,
            fipe_paid_ratio_tau=0.88,
            down_payment_pct=0.30,
        )

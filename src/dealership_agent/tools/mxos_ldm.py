"""Motor Determinístico MXOS LDM (Large Data Model) - Polo Curitiba.

Inferência algorítmica de liquidez, depreciação por KM (>15.000 km/ano),
propensão de aprovação de crédito bancário por faixa de entrada (<20% vs >40%)
e tempo de pátio (giro de estoque) em função do % FIPE pago na troca (80% a 92%).
Zero geração livre por LLM: garante precisão matemática sem alucinações.

Leitura e validação dos pesos, φ de km e spline de crédito vivem em
`mxos_ldm.weights` (porta única do JSON); aqui fica a regra de mesa de
avaliação. O caminho do JSON vem de `settings.ldm_weights_path` (relativo ao
diretório de trabalho — `/app` no Docker), não de `__file__`, que aponta para
o site-packages quando o pacote é instalado sem `-e`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from dealership_agent.config import settings
from mxos_ldm.weights import (
    CuritibaWeights,
    ModelCoefficients,
    credit_approval,
    km_liquidity_factor,
    load_weights,
)

HubType = Literal["linha_verde", "mario_tourinho"]
ModelName = Literal[
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
]
RecommendationType = Literal[
    "COMPRA_FORTE",
    "COMPRA_PADRAO",
    "RENEGOCIAR_TROCA",
    "REPASSE_B2B",
]

__all__ = ["CURITIBA_MODELS", "MXOSAutomotiveLDM", "ModelCoefficients", "load_ldm_weights"]


def curitiba_weights() -> CuritibaWeights:
    """Pesos validados (cacheados por caminho em `mxos_ldm.weights.load_weights`)."""
    return load_weights(settings.ldm_weights_path)


def load_ldm_weights() -> dict[str, Any]:
    """Matriz de pesos como dict (já validada contra o schema estrito)."""
    return curitiba_weights().model_dump()


# Fail-closed no import: sem pesos válidos, a simulação de financiamento não sobe.
CURITIBA_MODELS: dict[str, ModelCoefficients] = curitiba_weights().models_curitiba_top10


@dataclass(frozen=True)
class MXOSValuationResult:
    model: str
    hub: HubType
    annual_km: float
    km_liquidity_factor: float
    effective_liquidity_score: float
    fair_retail_price: float
    fipe_paid_ratio: float
    expected_days_on_lot: float
    net_expected_margin_pct: float
    credit_approval_prob: float
    effective_monthly_rate: float
    purchase_propensity_score: float
    recommendation: RecommendationType


class MXOSAutomotiveLDM:
    """Motor de inferência determinística para precificação, giro e crédito."""

    @staticmethod
    def compute_km_penalty(
        model: str, total_km: float, vehicle_age_years: float
    ) -> tuple[float, float]:
        """Retorna (km_anual, fator_liquidez_km no intervalo [0.150, 1.040])."""
        return km_liquidity_factor(curitiba_weights(), model, total_km, vehicle_age_years)

    @staticmethod
    def compute_credit_approval(
        down_payment_pct: float,
        credit_score: int = 650,
        vehicle_age_years: float = 3.0,
        dti_ratio: float = 0.28,
    ) -> tuple[float, float]:
        """Calcula (probabilidade_aprovacao, taxa_mensal_efetiva) via spline de LTV."""
        return credit_approval(
            curitiba_weights(), down_payment_pct, credit_score, vehicle_age_years, dti_ratio
        )

    @classmethod
    def evaluate_deal(
        cls,
        *,
        model: ModelName,
        fipe_value: float,
        total_km: float,
        vehicle_age_years: float,
        fipe_paid_ratio_tau: float,
        down_payment_pct: float,
        hub: HubType = "linha_verde",
        credit_score: int = 650,
        dti_ratio: float = 0.28,
    ) -> MXOSValuationResult:
        """Executa a inferência completa do LDM para uma operação de compra/troca + venda."""
        if fipe_value <= 0 or not math.isfinite(fipe_value):
            raise ValueError("fipe_value deve ser finito e positivo")

        coeff = CURITIBA_MODELS[model]
        hub_cfg = curitiba_weights().curitiba_hubs[hub]
        annual_km, phi_km = cls.compute_km_penalty(model, total_km, vehicle_age_years)

        # 1. Liquidez efetiva e Preço Justo de Varejo (VJM)
        elasticity = 0.31 if phi_km >= 1.0 else coeff.km_price_elasticity
        delta_km_price_discount = 1.0 - (phi_km**elasticity)
        fair_retail_price = (
            fipe_value * coeff.fipe_retail_retention_rho * (1.0 - delta_km_price_discount)
        )

        hub_match_factor = 1.0 if coeff.primary_hub in (hub, "ambos") else 0.91
        effective_liquidity = min(0.99, coeff.base_liquidity_L0 * phi_km * hub_match_factor)

        # 2. Giro de Estoque (Dias de Pátio) em função do % FIPE pago na troca (80% a 92%)
        tau = max(0.70, min(1.00, fipe_paid_ratio_tau))
        overpayment_tau = max(0.0, tau - coeff.optimal_trade_in_fipe_tau0)
        raw_days = (coeff.base_days_on_lot_D0 / (phi_km * hub_match_factor)) * math.exp(
            coeff.trade_in_stiffness_lambda * overpayment_tau
        )
        expected_days = round(min(180.0, raw_days), 1)

        # 3. Margem Líquida Esperada descontando preparação e custo financeiro de pátio
        prep_cost_pct = hub_cfg.avg_prep_cost_pct
        daily_holding_rate = hub_cfg.daily_holding_cost_rate
        acquisition_cost = fipe_value * tau
        prep_cost = fipe_value * prep_cost_pct
        holding_cost = acquisition_cost * daily_holding_rate * expected_days
        net_margin_pct = (
            fair_retail_price - acquisition_cost - prep_cost - holding_cost
        ) / fipe_value

        # 4. Propensão de Aprovação Bancária e Conversão Final do Lead
        credit_prob, eff_rate = cls.compute_credit_approval(
            down_payment_pct=down_payment_pct,
            credit_score=credit_score,
            vehicle_age_years=vehicle_age_years,
            dti_ratio=dti_ratio,
        )
        effective_credit_factor = 1.0 if down_payment_pct >= 0.99 else credit_prob
        propensity_score = round(
            0.55 * effective_credit_factor + 0.45 * effective_liquidity, 4
        )

        # 5. Regra de Decisão Determinística da Mesa de Avaliação
        if phi_km < 0.55:
            rec: RecommendationType = "REPASSE_B2B"
        elif net_margin_pct >= 0.095 and expected_days <= 25.0:
            rec = "COMPRA_FORTE"
        elif net_margin_pct >= 0.065 and expected_days <= 42.0:
            rec = "COMPRA_PADRAO"
        else:
            rec = "RENEGOCIAR_TROCA"

        return MXOSValuationResult(
            model=model,
            hub=hub,
            annual_km=annual_km,
            km_liquidity_factor=phi_km,
            effective_liquidity_score=round(effective_liquidity, 4),
            fair_retail_price=round(fair_retail_price, 2),
            fipe_paid_ratio=tau,
            expected_days_on_lot=expected_days,
            net_expected_margin_pct=round(net_margin_pct, 3),
            credit_approval_prob=credit_prob,
            effective_monthly_rate=eff_rate,
            purchase_propensity_score=propensity_score,
            recommendation=rec,
        )

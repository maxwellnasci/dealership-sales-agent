"""Matriz de pesos de Curitiba (`data/mxos_ldm_weights.json`, gerada pelo Antigravity).

Este módulo é a ÚNICA porta de leitura do arquivo: valida a estrutura exata
(strict, `extra="forbid"`) e expõe as duas fórmulas que a consomem, usadas
tanto pelo `LDMPredictor` quanto por `dealership_agent.tools.mxos_ldm` /
`simulate_financing`:

  - `km_liquidity_factor`: fator φ de liquidez por km/ano. Abaixo do benchmark
    (15 mil km/ano) dá bônus de até +4%; acima, decai exponencialmente com o
    γ do modelo e ganha um "penhasco" quadrático (η) acima do limiar de
    carro de aplicativo (25 mil km/ano).
  - `credit_approval`: probabilidade de aprovação bancária e taxa mensal
    efetiva, via spline de entrada (<20% / 20-40% / >40%).

Fail-closed: arquivo ausente ou malformado é erro explícito — quem chama
decide se segue sem pesos (o `LDMPredictor` aceita `weights=None`).
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_WEIGHTS_PATH = Path("data/mxos_ldm_weights.json")

HubName = Literal["linha_verde", "mario_tourinho"]


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)


class Metadata(_Strict):
    engine: Literal["mxos_ldm"]
    version: str
    region: str
    currency: Literal["BRL"]
    benchmark_km_per_year: float = Field(gt=0)
    app_usage_threshold_km_per_year: float = Field(gt=0)

    @model_validator(mode="after")
    def _limiar_app_acima_do_benchmark(self) -> Metadata:
        if self.app_usage_threshold_km_per_year <= self.benchmark_km_per_year:
            raise ValueError("app_usage_threshold_km_per_year deve exceder o benchmark")
        return self


class DownPaymentSpline(_Strict):
    low_ltv_cutoff: float = Field(gt=0, lt=1)
    high_ltv_cutoff: float = Field(gt=0, lt=1)
    sub_20_intercept: float
    sub_20_slope: float
    mid_20_40_intercept: float
    mid_20_40_slope: float
    over_40_intercept: float
    over_40_slope: float

    @model_validator(mode="after")
    def _cortes_ordenados(self) -> DownPaymentSpline:
        if self.low_ltv_cutoff >= self.high_ltv_cutoff:
            raise ValueError("low_ltv_cutoff deve ser menor que high_ltv_cutoff")
        return self


class RateSpread(_Strict):
    base_rate_am: float = Field(gt=0, lt=0.2)
    sub_20_penalty_am: float = Field(ge=0, lt=0.1)
    over_40_discount_am: float = Field(le=0, gt=-0.1)


class CreditScoringWeights(_Strict):
    beta_0_intercept: float
    beta_score_per_100pts: float
    beta_vehicle_age_over_3y: float
    beta_dti_over_30pct: float
    down_payment_spline: DownPaymentSpline
    rate_spread_adjustment_monthly: RateSpread


class HubConfig(_Strict):
    avg_prep_cost_pct: float = Field(ge=0, lt=1)
    min_target_margin_pct: float = Field(gt=0, lt=1)
    walk_in_financing_share: float = Field(ge=0, le=1)
    daily_holding_cost_rate: float = Field(ge=0, lt=0.1)


class ModelCoefficients(_Strict):
    segment: str
    primary_hub: Literal["linha_verde", "mario_tourinho", "ambos"]
    base_liquidity_L0: float = Field(gt=0, le=1)
    base_days_on_lot_D0: float = Field(gt=0)
    fipe_retail_retention_rho: float = Field(gt=0)
    optimal_trade_in_fipe_tau0: float = Field(gt=0, le=1.2)
    km_penalty_gamma: float = Field(ge=0)
    km_app_cliff_eta: float = Field(ge=0)
    trade_in_stiffness_lambda: float = Field(ge=0)
    km_price_elasticity: float = Field(default=0.402, gt=0)


class CuritibaWeights(_Strict):
    metadata: Metadata
    credit_scoring_weights: CreditScoringWeights
    curitiba_hubs: dict[HubName, HubConfig]
    models_curitiba_top10: dict[str, ModelCoefficients] = Field(min_length=1)

    @property
    def versao(self) -> str:
        return f"{self.metadata.engine}-{self.metadata.version}"


@lru_cache(maxsize=4)
def load_weights(path: Path | str = DEFAULT_WEIGHTS_PATH) -> CuritibaWeights:
    """Lê e valida o arquivo de pesos. Ausente ou malformado -> erro (fail-closed)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pesos do mxos_ldm não encontrados em {path.resolve()}")
    return CuritibaWeights.model_validate_json(path.read_text(encoding="utf-8"))


def km_liquidity_factor(
    weights: CuritibaWeights, model: str, total_km: float, vehicle_age_years: float
) -> tuple[float, float]:
    """(km_anual, φ) para um modelo do top 10. φ em [0.15, 1.04]."""
    coeff = weights.models_curitiba_top10[model]
    benchmark = weights.metadata.benchmark_km_per_year
    app_delta_k = (weights.metadata.app_usage_threshold_km_per_year - benchmark) / 1_000.0

    annual_km = total_km / max(0.5, float(vehicle_age_years))
    if annual_km <= benchmark:
        low_km_bonus = min(0.04, (benchmark - annual_km) / benchmark * 0.04)
        return round(annual_km, 1), round(1.0 + low_km_bonus, 3)

    delta_k = (annual_km - benchmark) / 1_000.0
    app_cliff = max(0.0, delta_k - app_delta_k) ** 2
    exponent = -(coeff.km_penalty_gamma * delta_k) - (coeff.km_app_cliff_eta * app_cliff)
    return round(annual_km, 1), round(max(0.15, math.exp(exponent)), 3)


def credit_approval(
    weights: CuritibaWeights,
    down_payment_pct: float,
    credit_score: int = 650,
    vehicle_age_years: float = 3.0,
    dti_ratio: float = 0.28,
) -> tuple[float, float]:
    """(probabilidade de aprovação 0-1, taxa mensal efetiva) via spline de entrada."""
    w = weights.credit_scoring_weights
    spline, spread = w.down_payment_spline, w.rate_spread_adjustment_monthly

    e = max(0.0, min(1.0, down_payment_pct))
    low_cut, high_cut = spline.low_ltv_cutoff, spline.high_ltv_cutoff
    if e < low_cut:
        g_e = spline.sub_20_intercept + spline.sub_20_slope * e
        rate_adj = spread.sub_20_penalty_am * ((low_cut - e) / low_cut)
    elif e <= high_cut:
        g_e = spline.mid_20_40_intercept + spline.mid_20_40_slope * (e - low_cut)
        rate_adj = 0.0
    else:
        g_e = spline.over_40_intercept + spline.over_40_slope * (e - high_cut)
        rate_adj = spread.over_40_discount_am * min(1.0, (e - high_cut) / 0.20)

    z = (
        w.beta_0_intercept
        + g_e
        + w.beta_score_per_100pts * ((credit_score - 600.0) / 100.0)
        - w.beta_vehicle_age_over_3y * max(0.0, vehicle_age_years - 3.0)
        - w.beta_dti_over_30pct * max(0.0, dti_ratio - 0.30)
    )
    prob = 1.0 / (1.0 + math.exp(-z))
    monthly_rate = max(
        0.0129, spread.base_rate_am + rate_adj - 0.0010 * ((credit_score - 650.0) / 150.0)
    )
    return round(prob, 3), round(monthly_rate, 4)


def faixa_entrada(weights: CuritibaWeights, down_payment_pct: float) -> str:
    """Rótulo da faixa de entrada do spline (ex: 'entrada<20%')."""
    spline = weights.credit_scoring_weights.down_payment_spline
    if down_payment_pct < spline.low_ltv_cutoff:
        return f"entrada<{spline.low_ltv_cutoff:.0%}"
    if down_payment_pct > spline.high_ltv_cutoff:
        return f"entrada>{spline.high_ltv_cutoff:.0%}"
    return f"entrada {spline.low_ltv_cutoff:.0%}-{spline.high_ltv_cutoff:.0%}"

"""Tool de simulação de financiamento (determinística).

Price → parcela pela fórmula de amortização Price (sistema francês/PRICE).
Nenhum número aqui passa pelo LLM: o agente só verbaliza o `FinancingQuote`
devolvido. Tornar isto uma função pura e testável é o que permite afirmar
"a parcela é R$ X" sem risco de alucinação.
"""

from __future__ import annotations

from dealership_agent.schemas.tools import FinancingQuote, Vehicle
from dealership_agent.tools.mxos_ldm import MXOSAutomotiveLDM
from mxos_ldm.data_engine import ANO_REFERENCIA

DEFAULT_MONTHLY_RATE = 0.0189  # 1,89% a.m. — taxa base de vitrine calibrada pelo LDM
DEFAULT_TERM_MONTHS = 48
DEFAULT_DOWN_PAYMENT_PCT = 0.20
REFERENCE_YEAR = ANO_REFERENCIA  # mesmo ano-base do mxos_ldm (idade do veículo)


def simulate_financing(
    vehicle: Vehicle,
    *,
    down_payment: float | None = None,
    term_months: int = DEFAULT_TERM_MONTHS,
    monthly_rate: float | None = None,
    credit_score: int = 650,
    dti_ratio: float = 0.28,
    reference_year: int = REFERENCE_YEAR,
) -> FinancingQuote:
    """Calcula a parcela pelo sistema PRICE conectada ao motor `mxos_ldm`.

    PMT = PV * i / (1 - (1 + i)^-n), onde PV é o valor financiado e `i`
    reflete `effective_monthly_rate` calibrada por faixa de entrada (<20% vs >40%),
    além de retornar `credit_approval_prob`.
    """
    if term_months <= 0:
        raise ValueError("term_months deve ser positivo")

    down = down_payment if down_payment is not None else vehicle.price * DEFAULT_DOWN_PAYMENT_PCT
    down = max(0.0, min(down, vehicle.price))
    principal = vehicle.price - down

    down_pct = (down / vehicle.price) if vehicle.price > 0 else DEFAULT_DOWN_PAYMENT_PCT
    vehicle_age = max(0.5, float(reference_year - vehicle.year))
    approval_prob, ldm_effective_rate = MXOSAutomotiveLDM.compute_credit_approval(
        down_payment_pct=down_pct,
        credit_score=credit_score,
        vehicle_age_years=vehicle_age,
        dti_ratio=dti_ratio,
    )
    effective_rate = ldm_effective_rate if monthly_rate is None else monthly_rate

    if principal <= 0:
        monthly = 0.0
    elif effective_rate == 0:
        monthly = principal / term_months
    else:
        factor = (1 + effective_rate) ** -term_months
        monthly = principal * effective_rate / (1 - factor)

    monthly = round(monthly, 2)
    total = round(monthly * term_months + down, 2)

    return FinancingQuote(
        vehicle_id=vehicle.vehicle_id,
        vehicle_label=vehicle.short_label(),
        price=vehicle.price,
        down_payment=round(down, 2),
        term_months=term_months,
        monthly_rate=effective_rate,
        monthly_payment=monthly,
        total_paid=total,
        credit_approval_prob=approval_prob,
    )

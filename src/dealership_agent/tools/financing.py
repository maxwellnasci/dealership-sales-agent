"""Tool de simulação de financiamento (determinística).

Price → parcela pela fórmula de amortização Price (sistema francês/PRICE).
Nenhum número aqui passa pelo LLM: o agente só verbaliza o `FinancingQuote`
devolvido. Tornar isto uma função pura e testável é o que permite afirmar
"a parcela é R$ X" sem risco de alucinação.
"""

from __future__ import annotations

from dealership_agent.schemas.tools import FinancingQuote, Vehicle

DEFAULT_MONTHLY_RATE = 0.019  # 1,9% a.m. — taxa de vitrine da revenda (mock)
DEFAULT_TERM_MONTHS = 48
DEFAULT_DOWN_PAYMENT_PCT = 0.20


def simulate_financing(
    vehicle: Vehicle,
    *,
    down_payment: float | None = None,
    term_months: int = DEFAULT_TERM_MONTHS,
    monthly_rate: float = DEFAULT_MONTHLY_RATE,
) -> FinancingQuote:
    """Calcula a parcela pelo sistema PRICE.

    PMT = PV * i / (1 - (1 + i)^-n), onde PV é o valor financiado.
    """
    if term_months <= 0:
        raise ValueError("term_months deve ser positivo")

    down = down_payment if down_payment is not None else vehicle.price * DEFAULT_DOWN_PAYMENT_PCT
    down = max(0.0, min(down, vehicle.price))
    principal = vehicle.price - down

    if principal <= 0:
        monthly = 0.0
    elif monthly_rate == 0:
        monthly = principal / term_months
    else:
        factor = (1 + monthly_rate) ** -term_months
        monthly = principal * monthly_rate / (1 - factor)

    monthly = round(monthly, 2)
    total = round(monthly * term_months + down, 2)

    return FinancingQuote(
        vehicle_id=vehicle.vehicle_id,
        vehicle_label=vehicle.short_label(),
        price=vehicle.price,
        down_payment=round(down, 2),
        term_months=term_months,
        monthly_rate=monthly_rate,
        monthly_payment=monthly,
        total_paid=total,
    )

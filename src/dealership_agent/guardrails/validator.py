"""Validador anti-alucinação (camada 2).

Camada 1 é estrutural: preços/veículos só nascem de tools tipadas. Camada 2 é
esta verificação da resposta final — antes de enviar ao lead, conferimos que
todo valor monetário citado no texto corresponde a um fato presente no estado
(preço de veículo recuperado, parcela/entrada/total da simulação). Se o texto
contém um número de dinheiro que não casa com nenhum fato conhecido, marcamos
a flag `unverified_amount`: o nó de síntese então reescreve sem o número
inventado, em vez de mentir um preço para o cliente.
"""

from __future__ import annotations

import re

from dealership_agent.schemas.tools import FinancingQuote, Vehicle

# Captura valores em reais: "R$ 89.900", "89900", "1.234,56", "89.900,00".
_MONEY_RE = re.compile(r"R?\$?\s*\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})?|\b\d{4,}\b")


def _normalize(token: str) -> float | None:
    cleaned = token.replace("R$", "").replace(" ", "").strip()
    if not cleaned:
        return None
    # Formato BR: ponto = milhar, vírgula = decimal.
    cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def allowed_amounts(
    vehicles: list[Vehicle], quote: FinancingQuote | None
) -> set[int]:
    """Conjunto de valores (arredondados a inteiro) que o agente PODE citar."""
    amounts: set[int] = {round(v.price) for v in vehicles}
    if quote is not None:
        amounts.update(
            {
                round(quote.price),
                round(quote.down_payment),
                round(quote.monthly_payment),
                round(quote.total_paid),
            }
        )
    return amounts


def validate_reply(
    reply: str, vehicles: list[Vehicle], quote: FinancingQuote | None
) -> list[str]:
    """Retorna a lista de flags encontradas (vazia = resposta confiável)."""
    allowed = allowed_amounts(vehicles, quote)
    flags: list[str] = []

    # Quilometragem não é preço: remove "<n> km" antes de procurar valores.
    scan_text = re.sub(r"\d[\d.\s]*km", " ", reply, flags=re.IGNORECASE)

    for match in _MONEY_RE.findall(scan_text):
        value = _normalize(match)
        if value is None or value < 1000:
            # Ignora anos, quantidades pequenas, prazos (ex: "48 meses").
            continue
        if not any(abs(value - a) <= 1 for a in allowed):
            flags.append(f"unverified_amount:{match.strip()}")

    return flags

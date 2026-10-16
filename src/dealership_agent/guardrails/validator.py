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

from dealership_agent.schemas.tools import FinancingQuote, TestDriveSlot, Vehicle

# Valores monetários. Alternativas ORDENADAS para não se sombrearem (o bug do
# regex antigo: `\d{1,3}...` casava "550" em "55000" e a alternativa de inteiro
# cru nunca era alcançada, deixando "55000" passar). Agora: primeiro com R$,
# depois com separador de milhar, por fim o inteiro cru de 4+ dígitos.
_MONEY_RE = re.compile(
    r"R\$\s*\d[\d.\s]*(?:,\d{2})?"
    r"|\b\d{1,3}(?:\.\d{3})+(?:,\d{2})?\b"
    r"|\b\d{4,}(?:,\d{2})?\b"
)
# "55 mil", "140 mil", "1,5 mil" — capturado à parte; o sufixo multiplica ×1000.
_MIL_RE = re.compile(r"\d+(?:[.,]\d+)?\s*mil\b", re.IGNORECASE)

# Captura datas "DD/MM" (com ou sem ano), para checar ambiguidade de agenda.
_DATE_RE = re.compile(r"\b(\d{2})/(\d{2})(?:/\d{2,4})?\b")


def _normalize(token: str) -> float | None:
    cleaned = token.replace("R$", "").replace("mil", "").replace(" ", "").strip()
    if not cleaned:
        return None
    # Formato BR: ponto = milhar, vírgula = decimal.
    cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_vehicle_year(raw: str, value: float) -> bool:
    """Inteiro cru de 4 dígitos em 1990-2099 (sem R$ nem separador) é ano de
    modelo (ex: 2022), não um preço — evita falso positivo do guardrail."""
    digits = re.sub(r"\D", "", raw)
    return (
        "R$" not in raw
        and "." not in raw
        and "," not in raw
        and len(digits) == 4
        and 1990 <= value <= 2099
    )


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
    reply: str,
    vehicles: list[Vehicle],
    quote: FinancingQuote | None,
    slot: TestDriveSlot | None = None,
) -> list[str]:
    """Retorna a lista de flags encontradas (vazia = resposta confiável).

    Checa valores monetários (contra estoque/simulação) e, em contexto de
    agendamento (`slot` presente), datas: qualquer DD/MM diferente do horário
    reservado vira `unverified_date` — evita o agente oferecer um segundo
    horário inventado e criar ambiguidade na agenda da revenda.
    """
    allowed = allowed_amounts(vehicles, quote)
    flags: list[str] = []

    # Quilometragem não é preço: remove "<n> km" antes de procurar valores.
    scan_text = re.sub(r"\d[\d.\s]*km", " ", reply, flags=re.IGNORECASE)

    def _check(raw: str, value: float) -> None:
        if not any(abs(value - a) <= 1 for a in allowed):
            flags.append(f"unverified_amount:{raw.strip()}")

    # "N mil" -> N*1000, capturado e removido antes para não recontar o número.
    for m in _MIL_RE.finditer(scan_text):
        base_val = _normalize(m.group(0))
        if base_val is not None:
            _check(m.group(0), base_val * 1000)
    scan_text = _MIL_RE.sub(" ", scan_text)

    for match in _MONEY_RE.findall(scan_text):
        value = _normalize(match)
        if value is None or value < 1000:
            # Ignora quantidades pequenas e prazos (ex: "48 meses").
            continue
        if _is_vehicle_year(match, value):
            continue  # ano de modelo, não preço
        _check(match, value)

    if slot is not None:
        allowed_date = f"{slot.starts_at:%d/%m}"
        seen: set[str] = set()
        for dd, mm in _DATE_RE.findall(reply):
            date = f"{dd}/{mm}"
            if date != allowed_date and date not in seen:
                seen.add(date)
                flags.append(f"unverified_date:{date}")

    return flags

"""Validador anti-alucinação (camada 2).

Camada 1 é estrutural: preços/veículos/horários só nascem de tools tipadas.
Camada 2 é esta verificação da resposta final — antes de enviar ao lead,
conferimos que cada fato citado no texto pertence à ENTIDADE a que está
amarrado, e não apenas que "existe em algum lugar do estado".

Validação amarrada por entidade (cross-entity):

  - O texto é fatiado por frase e, dentro dela, a cada menção de modelo
    ("Corolla", "Polo"). Preço, ano e km de um segmento têm de ser fatos DAQUELE
    veículo (preço de estoque, ou entrada/parcela/total da simulação dele).
    Colar o preço do Polo no Corolla é barrado, mesmo o valor existindo.
  - Dono de cada trecho: o modelo citado logo antes; o começo de uma frase
    pertence ao 1º modelo da própria frase ("Por R$ X você leva o Corolla");
    frase sem modelo herda o último citado. Só o que vem antes de QUALQUER
    menção não tem dono: ali basta o valor/km ser um fato conhecido do turno.
  - Modelo do catálogo que não está entre os fatos do turno vira
    `unverified_vehicle` (o agente não pode "trazer" carro que não buscou).
  - Com test-drive reservado (`slot`), data, horário e dia da semana têm de ser
    os do slot, e citados junto do veículo do slot.

Qualquer flag faz o nó de síntese cair no rascunho factual. A heurística erra
para o lado seguro: construções como "Corolla e Polo saem por X e Y" (fato fora
do segmento do dono) viram falso positivo -> resposta mais seca, nunca mentira.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from dealership_agent.schemas.tools import FinancingQuote, TestDriveSlot, Vehicle
from dealership_agent.tools.inventory import InventoryUnavailable, load_catalog

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
# Quilometragem: "38.000 km", "38000km", "38 mil km". Extraída ANTES do
# dinheiro (para "38 mil km" não virar R$ 38.000) e conferida com o veículo.
_KM_RE = re.compile(r"\b(\d[\d.]*(?:,\d+)?)\s*(mil\s*)?km\b", re.IGNORECASE)

# Agenda. Datas "DD/MM" (com ou sem ano); horários "09h00", "9h", "16 h 30",
# "09:00", "9 horas"; "às 4 da tarde"; dia da semana por extenso.
_DATE_RE = re.compile(r"\b(\d{2})/(\d{2})(?:/\d{2,4})?\b")
_TIME_RE = re.compile(
    r"\b(\d{1,2})\s*(?:horas?\b|h(?![a-zà-ú])|:(?=\d{2}))\s*(\d{2})?(?!\d)",
    re.IGNORECASE,
)
_AS_HOUR_RE = re.compile(r"\bàs\s+(\d{1,2})\b(?:\s+da\s+(tarde|noite))?", re.IGNORECASE)
_WEEKDAY_RE = re.compile(
    r"\b(segunda|terça|terca|quarta|quinta|sexta)(?:-|\s)feira\b|\b(sábado|sabado|domingo)\b",
    re.IGNORECASE,
)
# Fim de frase: pontuação + espaço, ou quebra de linha. "129.900" não quebra.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WEEKDAYS = {
    "segunda": 0, "terça": 1, "terca": 1, "quarta": 2, "quinta": 3,
    "sexta": 4, "sábado": 5, "sabado": 5, "domingo": 6,
}

Flag = Callable[[str, str], None]


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


def _quote_amounts(quote: FinancingQuote) -> set[int]:
    return {
        round(quote.price),
        round(quote.down_payment),
        round(quote.monthly_payment),
        round(quote.total_paid),
    }


def allowed_amounts(
    vehicles: list[Vehicle], quote: FinancingQuote | None
) -> set[int]:
    """Conjunto de valores (arredondados a inteiro) que o agente PODE citar
    em texto sem dono (antes de qualquer menção de modelo)."""
    amounts: set[int] = {round(v.price) for v in vehicles}
    if quote is not None:
        amounts.update(_quote_amounts(quote))
    return amounts


@dataclass
class _Entity:
    """Fatos de UM modelo citado: só estes valem dentro do segmento dele."""

    label: str
    vehicle_ids: set[str] = field(default_factory=set)
    amounts: set[int] = field(default_factory=set)
    years: set[int] = field(default_factory=set)
    mileages: set[int] = field(default_factory=set)

    @property
    def known(self) -> bool:
        return bool(self.vehicle_ids)


def _model_key(text: str) -> str:
    """'HR-V', 'hr v', 'HRV' -> 'hrv' (chave de comparação de modelo)."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _model_pattern(model: str) -> str:
    parts = re.findall(r"[a-z0-9]+", model.lower())
    return r"[-\s]?".join(re.escape(p) for p in parts)


def _catalog() -> list[Vehicle]:
    # O catálogo só enriquece a detecção de modelos; indisponível -> segue sem.
    try:
        return load_catalog()
    except InventoryUnavailable:
        return []


class _Facts:
    """Fatos verificáveis do turno, indexados por modelo."""

    def __init__(
        self,
        vehicles: list[Vehicle],
        quote: FinancingQuote | None,
        slot: TestDriveSlot | None,
    ) -> None:
        catalog = _catalog()
        by_id = {v.vehicle_id: v for v in catalog}
        known = {v.vehicle_id: v for v in vehicles}
        # Veículo da simulação/agendamento é fato do turno mesmo que a lista
        # de recuperados não o traga.
        for ref in (quote, slot):
            if ref is not None and ref.vehicle_id not in known and ref.vehicle_id in by_id:
                known[ref.vehicle_id] = by_id[ref.vehicle_id]

        self.slot = slot
        self.global_amounts = allowed_amounts(list(known.values()), quote)
        self.global_mileages = {v.mileage_km for v in known.values()}

        self.entities: dict[str, _Entity] = {}
        for v in catalog:
            key = _model_key(v.model)
            self.entities.setdefault(key, _Entity(label=v.model))
        for v in known.values():
            ent = self.entities.setdefault(_model_key(v.model), _Entity(label=v.model))
            ent.vehicle_ids.add(v.vehicle_id)
            ent.amounts.add(round(v.price))
            ent.years.add(v.year)
            ent.mileages.add(v.mileage_km)
            if quote is not None and quote.vehicle_id == v.vehicle_id:
                ent.amounts.update(_quote_amounts(quote))

        patterns = sorted(
            {_model_pattern(e.label) for e in self.entities.values()}, key=len, reverse=True
        )
        self.mention_re = (
            re.compile(rf"(?<![\w-])(?:{'|'.join(patterns)})(?![\w-])", re.IGNORECASE)
            if patterns
            else None
        )

    def segments(self, reply: str) -> list[tuple[_Entity | None, str]]:
        """Fatia o texto em (dono, trecho), com as regras de dono do módulo."""
        if self.mention_re is None:
            return [(None, reply)]
        out: list[tuple[_Entity | None, str]] = []
        owner: _Entity | None = None
        for sentence in _SENTENCE_RE.split(reply):
            mentions = list(self.mention_re.finditer(sentence))
            if not mentions:
                out.append((owner, sentence))
                continue
            first = self.entities[_model_key(mentions[0].group(0))]
            out.append((first, sentence[: mentions[0].start()]))
            for i, m in enumerate(mentions):
                owner = self.entities[_model_key(m.group(0))]
                end = mentions[i + 1].start() if i + 1 < len(mentions) else len(sentence)
                out.append((owner, sentence[m.end() : end]))
        return out


def _check_mileage(chunk: str, owner: _Entity | None, facts: _Facts, flag: Flag) -> str:
    allowed = owner.mileages if owner is not None else facts.global_mileages
    for m in _KM_RE.finditer(chunk):
        value = _normalize(m.group(1))
        if value is None:
            continue
        if m.group(2):
            value *= 1000
        if not any(abs(value - a) <= 1 for a in allowed):
            flag("unverified_mileage", m.group(0).strip())
    return _KM_RE.sub(" ", chunk)


def _check_schedule(chunk: str, owner: _Entity | None, facts: _Facts, flag: Flag) -> str:
    """Datas/horários sempre saem do texto (não são preço nem ano); só são
    conferidos quando há slot reservado — sem slot, evita falso positivo em
    conversa comum."""
    slot = facts.slot
    mentioned = False

    for dd, mm in _DATE_RE.findall(chunk):
        mentioned = True
        if slot is not None and f"{dd}/{mm}" != f"{slot.starts_at:%d/%m}":
            flag("unverified_date", f"{dd}/{mm}")
    chunk = _DATE_RE.sub(" ", chunk)

    times: list[tuple[int, int]] = []
    for m in _TIME_RE.finditer(chunk):
        times.append((int(m.group(1)), int(m.group(2) or 0)))
    chunk = _TIME_RE.sub(" ", chunk)
    for m in _AS_HOUR_RE.finditer(chunk):
        hour = int(m.group(1))
        if m.group(2) and hour < 12:
            hour += 12
        times.append((hour, 0))
    chunk = _AS_HOUR_RE.sub(" ", chunk)
    for hour, minute in times:
        if hour > 23 or minute > 59:
            continue  # "48h" de prazo etc.: não é hora do dia
        mentioned = True
        if slot is not None and (hour, minute) != (slot.starts_at.hour, slot.starts_at.minute):
            flag("unverified_time", f"{hour:02d}h{minute:02d}")

    for m in _WEEKDAY_RE.finditer(chunk):
        mentioned = True
        day = _WEEKDAYS[(m.group(1) or m.group(2)).lower()]
        if slot is not None and day != slot.starts_at.weekday():
            flag("unverified_weekday", m.group(0))
    chunk = _WEEKDAY_RE.sub(" ", chunk)

    if (
        slot is not None
        and mentioned
        and owner is not None
        and slot.vehicle_id not in owner.vehicle_ids
    ):
        flag("unverified_slot_vehicle", owner.label)
    return chunk


def _check_amounts_and_years(
    chunk: str, owner: _Entity | None, facts: _Facts, flag: Flag
) -> None:
    allowed = owner.amounts if owner is not None else facts.global_amounts

    def _check(raw: str, value: float) -> None:
        if not any(abs(value - a) <= 1 for a in allowed):
            flag("unverified_amount", raw.strip().rstrip("."))

    # "N mil" -> N*1000, capturado e removido antes para não recontar o número.
    for m in _MIL_RE.finditer(chunk):
        base_val = _normalize(m.group(0))
        if base_val is not None:
            _check(m.group(0), base_val * 1000)
    chunk = _MIL_RE.sub(" ", chunk)

    for match in _MONEY_RE.findall(chunk):
        value = _normalize(match)
        if value is None or value < 1000:
            # Ignora quantidades pequenas e prazos (ex: "48 meses").
            continue
        if _is_vehicle_year(match, value):
            # Ano solto ("temos 2019 e 2022") não tem dono; colado num modelo
            # tem de ser o ano DAQUELE veículo.
            if owner is not None and int(value) not in owner.years:
                flag("unverified_year", match.strip())
            continue
        _check(match, value)


def validate_reply(
    reply: str,
    vehicles: list[Vehicle],
    quote: FinancingQuote | None,
    slot: TestDriveSlot | None = None,
) -> list[str]:
    """Retorna a lista de flags encontradas (vazia = resposta confiável).

    Flags: `unverified_amount`, `unverified_year`, `unverified_mileage`,
    `unverified_vehicle` e, com `slot`, `unverified_date`, `unverified_time`,
    `unverified_weekday`, `unverified_slot_vehicle`.
    """
    facts = _Facts(vehicles, quote, slot)
    flags: list[str] = []

    def flag(kind: str, raw: str) -> None:
        item = f"{kind}:{raw}"
        if item not in flags:
            flags.append(item)

    for owner, chunk in facts.segments(reply):
        if owner is not None and not owner.known:
            flag("unverified_vehicle", owner.label)
        chunk = _check_mileage(chunk, owner, facts, flag)
        chunk = _check_schedule(chunk, owner, facts, flag)
        _check_amounts_and_years(chunk, owner, facts, flag)

    return flags

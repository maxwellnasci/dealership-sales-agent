"""Ponte LDM -> vendedor humano: o Card Executivo das 08h30.

Recebe o State do LangGraph (um `Mapping`, como o `GraphState`) e devolve um
`SellerCard` com o texto pronto para envio e o horário agendado (próximas
08h30 em America/Sao_Paulo).

Chaves do State lidas aqui (todas opcionais, além do `lead_profile` existente):
  - `ldm_lead`       : LeadData ou dict equivalente
  - `ldm_vehicle`    : VehicleAppraisalInput ou dict (carro da troca)
  - `ldm_prediction` : PredictionOutput já calculada (senão, usa `predictor`)

Fail-closed: sem FIPE válida, sem carro da troca ou com dado inválido, o card
sai BLOQUEADO — com o motivo e sem nenhum número de avaliação. O vendedor
nunca recebe um teto de avaliação que o modelo não pôde calcular.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, ValidationError

from mxos_ldm.model import LDMPredictor, ensure_fipe
from mxos_ldm.schemas import (
    FipeIndisponivelError,
    LeadData,
    PredictionOutput,
    VehicleAppraisalInput,
)

TZ_CURITIBA = ZoneInfo("America/Sao_Paulo")
HORARIO_ENVIO = time(8, 30)

PRIORIDADE_ALTA = 60.0
PRIORIDADE_MEDIA = 35.0

Prioridade = Literal["alta", "media", "baixa", "bloqueado"]


class SellerCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    lead_id: str
    bloqueado: bool
    motivo_bloqueio: str | None
    prioridade: Prioridade
    texto: str
    envio_agendado_para: datetime


def proximo_envio(now: datetime) -> datetime:
    """Próximas 08h30 em Curitiba estritamente depois de `now` (naive = horário local)."""
    local = now.replace(tzinfo=TZ_CURITIBA) if now.tzinfo is None else now.astimezone(TZ_CURITIBA)
    alvo = datetime.combine(local.date(), HORARIO_ENVIO, tzinfo=TZ_CURITIBA)
    if local >= alvo:
        alvo = datetime.combine(local.date() + timedelta(days=1), HORARIO_ENVIO, tzinfo=TZ_CURITIBA)
    return alvo


def formatar_reais(valor: float) -> str:
    """1234567.8 -> 'R$ 1.234.567,80'."""
    s = f"{valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"R$ {s}"


def formatar_numero(valor: float, casas: int = 1) -> str:
    """76.6 -> '76,6' (vírgula decimal brasileira)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def formatar_pct(valor: float, casas: int = 1) -> str:
    """76.6 -> '76,6%'."""
    return f"{formatar_numero(valor, casas)}%"


def classificar_prioridade(probabilidade: float) -> Prioridade:
    if probabilidade >= PRIORIDADE_ALTA:
        return "alta"
    if probabilidade >= PRIORIDADE_MEDIA:
        return "media"
    return "baixa"


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump()
    return dict(value)


def build_seller_card(
    state: Mapping[str, Any],
    *,
    predictor: LDMPredictor | None = None,
    now: datetime | None = None,
) -> SellerCard:
    """Monta o Card Executivo do Vendedor a partir do State do grafo."""
    envio = proximo_envio(now or datetime.now(TZ_CURITIBA))
    profile = state.get("lead_profile")
    lead_raw = _as_dict(state.get("ldm_lead")) or {}
    lead_id = str(lead_raw.get("lead_id") or getattr(profile, "phone", None) or "desconhecido")
    nome = lead_raw.get("nome") or getattr(profile, "name", None) or "Lead"

    def bloqueado(motivo: str) -> SellerCard:
        texto = (
            f"🚫 CARD BLOQUEADO — {nome} ({lead_id})\n"
            f"Motivo: {motivo}\n"
            "Ação: confirmar os dados do carro da troca antes de qualquer proposta."
        )
        return SellerCard(
            lead_id=lead_id,
            bloqueado=True,
            motivo_bloqueio=motivo,
            prioridade="bloqueado",
            texto=texto,
            envio_agendado_para=envio,
        )

    vehicle_raw = _as_dict(state.get("ldm_vehicle"))
    if vehicle_raw is None:
        return bloqueado("sem dados do carro da troca")
    try:
        ensure_fipe(vehicle_raw)
        vehicle = VehicleAppraisalInput.model_validate(vehicle_raw)
        lead = LeadData.model_validate(lead_raw)
    except FipeIndisponivelError:
        return bloqueado("FIPE indisponível — avaliação não calculada")
    except ValidationError as exc:
        campos = ", ".join(sorted({str(e["loc"][0]) for e in exc.errors()}))
        return bloqueado(f"dados inválidos ({campos})")

    prediction = state.get("ldm_prediction")
    if prediction is None and predictor is not None:
        prediction = predictor.predict(lead, vehicle)
    if prediction is None:
        return bloqueado("previsão do LDM indisponível")
    prediction = PredictionOutput.model_validate(_as_dict(prediction))

    prioridade = classificar_prioridade(prediction.probabilidade_fechamento)
    pedido_pct = vehicle.percentual_fipe_pedido * 100.0
    teto_pct = prediction.avaliacao_fipe_recomendada
    if prediction.gap_pedido_pp > 0:
        negociacao = (
            f"Cliente pede {formatar_pct(pedido_pct)} da FIPE; teto é {formatar_pct(teto_pct)}"
            f" → negociar {formatar_numero(prediction.gap_pedido_pp)} p.p. para baixo."
        )
    else:
        negociacao = (
            f"Pedido de {formatar_pct(pedido_pct)} da FIPE está dentro do teto "
            f"({formatar_pct(teto_pct)}) — pode fechar a avaliação."
        )

    linhas = [
        f"📋 CARD EXECUTIVO — {envio:%d/%m %H:%M}",
        f"Lead: {nome} ({lead_id})",
    ]
    if lead.veiculo_interesse:
        linhas.append(f"Interesse: {lead.veiculo_interesse}")
    linhas += [
        f"Prioridade: {prioridade.upper()} — "
        f"{formatar_pct(prediction.probabilidade_fechamento)} de fechamento",
        f"Entrada: {formatar_pct(lead.valor_entrada_pct * 100, 0)} do valor",
        f"Troca: {vehicle.descricao or f'ano {vehicle.ano_modelo}'} · "
        f"{vehicle.km_rodados:,} km".replace(",", ".")
        + f" · FIPE {formatar_reais(vehicle.valor_fipe)}",
        f"Giro estimado da troca: {prediction.giro_estimado_dias} dias",
        f"Avaliação máxima: {formatar_reais(prediction.valor_avaliacao_max)} "
        f"({formatar_pct(teto_pct)} FIPE, margem líquida "
        f"{formatar_pct(prediction.margem_liquida_alvo * 100, 0)})",
        negociacao,
    ]
    if prediction.credit_approval_prob is not None:
        linhas.append(
            f"Aprovação de crédito estimada: {formatar_pct(prediction.credit_approval_prob * 100)}"
        )
    if prediction.ajustes_liquidez:
        linhas.append(
            f"Ajustes de liquidez ({prediction.versao_pesos}): "
            + ", ".join(prediction.ajustes_liquidez)
        )
    return SellerCard(
        lead_id=lead_id,
        bloqueado=False,
        motivo_bloqueio=None,
        prioridade=prioridade,
        texto="\n".join(linhas),
        envio_agendado_para=envio,
    )

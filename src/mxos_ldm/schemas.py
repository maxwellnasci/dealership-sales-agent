"""Contratos tipados do LDM.

Todos os modelos são `strict` (sem coerção silenciosa de "85000" -> 85000.0),
`extra="forbid"` (campo desconhecido é erro, não ruído ignorado) e `frozen`
(uma previsão emitida não muda depois). NaN/inf são rejeitados em todo float.

FIPE é o dado crítico: sem ela não existe avaliação. `valor_fipe` é obrigatório
e positivo — `None`, NaN, zero ou negativo viram `ValidationError`, e quem chama
pelo caminho de alto nível (`LDMPredictor.predict_raw`, `sales_bridge`) recebe
`FipeIndisponivelError`: fail-closed, nunca um número inventado.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

_ANO_MIN = 1990
_ANO_MAX = date.today().year + 1  # ano-modelo pode ser o próximo


class FipeIndisponivelError(ValueError):
    """FIPE nula/inválida: a avaliação é bloqueada (fail-closed)."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)


class LeadData(_StrictModel):
    """Dados do lead relevantes para a probabilidade de fechamento."""

    lead_id: str = Field(min_length=1, description="Identificador do lead (telefone/thread_id).")
    nome: str | None = Field(default=None, description="Primeiro nome, se conhecido.")
    valor_entrada_pct: float = Field(
        ge=0.0, le=1.0, description="Entrada como fração do preço do carro desejado (0-1)."
    )
    veiculo_interesse: str | None = Field(
        default=None, description="Rótulo do veículo de interesse (ex: 'Toyota Corolla 2022')."
    )


class VehicleAppraisalInput(_StrictModel):
    """O carro que o lead oferece na troca."""

    ano_modelo: int = Field(ge=_ANO_MIN, le=_ANO_MAX)
    km_rodados: int = Field(ge=0, le=1_500_000)
    valor_fipe: float = Field(gt=0.0, description="Valor FIPE de referência, em reais.")
    percentual_fipe_pedido: float = Field(
        gt=0.0, le=2.0, description="Pedido do cliente pelo carro, em fração da FIPE (1.0=FIPE)."
    )
    descricao: str | None = Field(default=None, description="Ex: 'VW Gol 1.0 2018'.")
    modelo: str | None = Field(
        default=None, description="Modelo no top 10 de Curitiba (ex: 'Onix') para o φ de km."
    )


class PredictionOutput(_StrictModel):
    """Saída do LDM para um par (lead, carro da troca)."""

    probabilidade_fechamento: float = Field(ge=0.0, le=100.0)
    giro_estimado_dias: int = Field(ge=1)
    avaliacao_fipe_recomendada: float = Field(
        ge=0.0, le=100.0, description="Teto da avaliação em % da FIPE para a margem alvo."
    )
    valor_avaliacao_max: float = Field(ge=0.0, description="Teto da avaliação em reais.")
    margem_liquida_alvo: float = Field(gt=0.0, lt=1.0)
    gap_pedido_pp: float = Field(
        description="Pedido do cliente menos o teto recomendado, em pontos percentuais de FIPE."
    )
    backend: str
    versao_pesos: str = "neutro"
    ajustes_liquidez: list[str] = Field(
        default_factory=list, description="Pesos de Curitiba aplicados (φ de km, faixa)."
    )
    km_liquidity_factor: float | None = Field(default=None, ge=0.0, le=2.0)
    credit_approval_prob: float | None = Field(default=None, ge=0.0, le=1.0)

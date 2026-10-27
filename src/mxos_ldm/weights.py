"""Pesos de liquidez da Matriz de Curitiba (sobreposição de negócio ao modelo).

O modelo aprendido dá a probabilidade de fechamento e o giro "de base"; estes
pesos aplicam por cima as regras de liquidez da praça, de forma auditável:

  - km/ano acima de `km_ano.limite_km_ano` (padrão 15 mil): multiplica
    probabilidade e giro.
  - entrada abaixo de `faixas_entrada.limite_baixa` (20%) ou acima de
    `faixas_entrada.limite_alta` (40%): multiplica probabilidade e giro.

Formato do arquivo (`data/mxos_ldm_weights.json`):

    {
      "versao": "curitiba-2026-10",
      "km_ano": {"limite_km_ano": 15000,
                 "multiplicador_prob_acima": 0.9, "multiplicador_giro_acima": 1.2},
      "faixas_entrada": {"limite_baixa": 0.20, "limite_alta": 0.40,
                         "multiplicador_prob_baixa": 0.85, "multiplicador_prob_alta": 1.15,
                         "multiplicador_giro_baixa": 1.0, "multiplicador_giro_alta": 1.0}
    }

Fail-closed: arquivo presente mas malformado é erro (não cai silenciosamente
no neutro). Sem arquivo, `LiquidityWeights()` é neutro (todos os
multiplicadores = 1.0) e o modelo sai sem ajuste.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_WEIGHTS_PATH = Path("data/mxos_ldm_weights.json")


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)


class KmAnoWeights(_Strict):
    limite_km_ano: float = Field(default=15_000.0, gt=0.0)
    multiplicador_prob_acima: float = Field(default=1.0, gt=0.0, le=5.0)
    multiplicador_giro_acima: float = Field(default=1.0, gt=0.0, le=5.0)


class FaixasEntradaWeights(_Strict):
    limite_baixa: float = Field(default=0.20, ge=0.0, le=1.0)
    limite_alta: float = Field(default=0.40, ge=0.0, le=1.0)
    multiplicador_prob_baixa: float = Field(default=1.0, gt=0.0, le=5.0)
    multiplicador_prob_alta: float = Field(default=1.0, gt=0.0, le=5.0)
    multiplicador_giro_baixa: float = Field(default=1.0, gt=0.0, le=5.0)
    multiplicador_giro_alta: float = Field(default=1.0, gt=0.0, le=5.0)

    @model_validator(mode="after")
    def _faixas_ordenadas(self) -> FaixasEntradaWeights:
        if self.limite_baixa >= self.limite_alta:
            raise ValueError("limite_baixa deve ser menor que limite_alta")
        return self


class LiquidityWeights(_Strict):
    versao: str = "neutro"
    km_ano: KmAnoWeights = KmAnoWeights()
    faixas_entrada: FaixasEntradaWeights = FaixasEntradaWeights()

    def multiplicadores(
        self, km_ano: float, entrada_pct: float
    ) -> tuple[float, float, list[str]]:
        """(mult. probabilidade, mult. giro, ajustes aplicados) para um negócio.

        Uma regra só entra em `ajustes` se de fato mexe em algo (multiplicador
        != 1): com pesos neutros o card não anuncia ajuste que não houve.
        """
        prob, giro, ajustes = 1.0, 1.0, []
        regras: list[tuple[bool, float, float, str]] = []
        km, faixas = self.km_ano, self.faixas_entrada
        regras.append((
            km_ano > km.limite_km_ano,
            km.multiplicador_prob_acima,
            km.multiplicador_giro_acima,
            f"km_ano>{km.limite_km_ano:.0f}",
        ))
        regras.append((
            entrada_pct < faixas.limite_baixa,
            faixas.multiplicador_prob_baixa,
            faixas.multiplicador_giro_baixa,
            f"entrada<{faixas.limite_baixa:.0%}",
        ))
        regras.append((
            entrada_pct > faixas.limite_alta,
            faixas.multiplicador_prob_alta,
            faixas.multiplicador_giro_alta,
            f"entrada>{faixas.limite_alta:.0%}",
        ))
        for ativa, m_prob, m_giro, rotulo in regras:
            if ativa and (m_prob != 1.0 or m_giro != 1.0):
                prob *= m_prob
                giro *= m_giro
                ajustes.append(rotulo)
        return prob, giro, ajustes


def load_weights(path: Path | str = DEFAULT_WEIGHTS_PATH) -> LiquidityWeights:
    """Lê e valida o arquivo de pesos. Ausente -> neutro; malformado -> erro."""
    path = Path(path)
    if not path.exists():
        return LiquidityWeights()
    return LiquidityWeights.model_validate_json(path.read_text(encoding="utf-8"))

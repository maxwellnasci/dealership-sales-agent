"""Pipeline preditivo do LDM: fechamento, giro e teto de avaliação.

Dois modelos aprendidos + uma conta determinística:

  a) probabilidade_fechamento — classificador sobre (ano, km, FIPE, % pedido,
     % entrada). Saída em 0-100.
  b) giro_estimado_dias — regressor do tempo de pátio do carro da troca,
     treinado em log1p(dias) por causa da cauda longa dos encalhados.
  c) avaliacao_fipe_recomendada — NÃO é aprendida: é a conta fechada de margem
     em `avaliacao_fipe_recomendada()`, alimentada pelo giro previsto. Número
     de dinheiro sai de fórmula auditável, não de caixa-preta.

Por cima dos modelos aprendidos entram os pesos de liquidez da Matriz de
Curitiba (`weights.py`): multiplicadores de km/ano e de faixa de entrada sobre
a probabilidade e o giro — o teto de avaliação já sai do giro ajustado.

Backends: "xgboost", "sklearn" (HistGradientBoosting), "tabpfn" (opcional,
`pip install tabpfn`) ou "auto" (xgboost se instalado, senão sklearn).
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

from mxos_ldm.data_engine import (
    ANO_REFERENCIA,
    FEATURES_FECHAMENTO,
    FEATURES_GIRO,
    SyntheticDataset,
    validate_dataset,
)
from mxos_ldm.schemas import (
    FipeIndisponivelError,
    LeadData,
    PredictionOutput,
    VehicleAppraisalInput,
)
from mxos_ldm.weights import LiquidityWeights

Backend = Literal["auto", "xgboost", "sklearn", "tabpfn"]

MARGEM_LIQUIDA_ALVO = 0.12
TABPFN_MAX_AMOSTRAS = 10_000


@dataclass(frozen=True)
class CustosRevenda:
    """Premissas de custo da revenda, todas em fração da FIPE do carro da troca."""

    preco_revenda_pct_fipe: float = 1.00     # revende ~na FIPE no varejo
    depreciacao_diaria_pct: float = 0.0005   # ~1,5% a cada 30 dias de pátio
    custo_preparacao_pct: float = 0.025      # higienização, revisão, pequenos reparos
    custo_capital_diario_pct: float = 0.0004  # custo financeiro do carro parado


def avaliacao_fipe_recomendada(
    giro_dias: float,
    margem_alvo: float = MARGEM_LIQUIDA_ALVO,
    custos: CustosRevenda = CustosRevenda(),  # noqa: B008 - dataclass imutável
) -> float:
    """Teto de avaliação, em % da FIPE, que preserva `margem_alvo` líquida.

    Com tudo em fração da FIPE:
        revenda   = preco_revenda - depreciacao_diaria * giro
        custos    = preparacao + capital_diario * giro
        margem    = (revenda - aquisicao - custos) / revenda
    Resolvendo margem == margem_alvo para a aquisição:
        aquisicao = revenda * (1 - margem_alvo) - custos

    Resultado truncado em [0, 100]: 0 significa "não compre esse carro".
    """
    if not math.isfinite(giro_dias) or giro_dias < 0:
        raise ValueError("giro_dias deve ser finito e >= 0")
    if not 0.0 < margem_alvo < 1.0:
        raise ValueError("margem_alvo deve estar em (0, 1)")
    revenda = custos.preco_revenda_pct_fipe - custos.depreciacao_diaria_pct * giro_dias
    custo_total = custos.custo_preparacao_pct + custos.custo_capital_diario_pct * giro_dias
    teto = revenda * (1.0 - margem_alvo) - custo_total
    return round(min(max(teto, 0.0), 1.0) * 100.0, 2)


def margem_liquida(
    aquisicao_pct_fipe: float, giro_dias: float, custos: CustosRevenda = CustosRevenda()  # noqa: B008
) -> float:
    """Margem líquida (fração) de comprar a `aquisicao_pct_fipe` e girar em `giro_dias`."""
    revenda = custos.preco_revenda_pct_fipe - custos.depreciacao_diaria_pct * giro_dias
    custo_total = custos.custo_preparacao_pct + custos.custo_capital_diario_pct * giro_dias
    return (revenda - aquisicao_pct_fipe - custo_total) / revenda


def _xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def _load_tabpfn() -> tuple[type, type]:
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError as exc:
        raise ImportError(
            "backend 'tabpfn' requer o pacote opcional: "
            "pip install 'dealership-sales-agent[tabpfn]'"
        ) from exc
    return TabPFNClassifier, TabPFNRegressor


def _build_estimators(backend: Backend, seed: int) -> tuple[str, Any, Any]:
    if backend == "auto":
        backend = "xgboost" if _xgboost_available() else "sklearn"
    if backend == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        common = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "random_state": seed,
        }
        return "xgboost", XGBClassifier(eval_metric="logloss", **common), XGBRegressor(**common)
    if backend == "sklearn":
        return (
            "sklearn",
            HistGradientBoostingClassifier(random_state=seed),
            HistGradientBoostingRegressor(random_state=seed),
        )
    if backend == "tabpfn":
        clf_cls, reg_cls = _load_tabpfn()
        return "tabpfn", clf_cls(random_state=seed), reg_cls(random_state=seed)
    raise ValueError(f"backend desconhecido: {backend!r}")


class LDMPredictor:
    """Treina e serve as três previsões do LDM."""

    def __init__(
        self,
        backend: Backend = "auto",
        seed: int = 42,
        margem_alvo: float = MARGEM_LIQUIDA_ALVO,
        custos: CustosRevenda = CustosRevenda(),  # noqa: B008
        weights: LiquidityWeights = LiquidityWeights(),  # noqa: B008
    ) -> None:
        self.backend, self._clf, self._reg = _build_estimators(backend, seed)
        self.seed = seed
        self.margem_alvo = margem_alvo
        self.custos = custos
        self.weights = weights
        self._fitted = False

    def fit(self, ds: SyntheticDataset, holdout: float = 0.2) -> dict[str, float]:
        """Treina os dois modelos e devolve métricas no holdout.

        `mae_giro_baseline_dias` é o MAE de prever a mediana do treino para
        todo carro — o modelo de giro só agrega valor se ficar abaixo dele.
        """
        validate_dataset(ds)
        x_clf = ds.matrix(FEATURES_FECHAMENTO)
        x_reg = ds.matrix(FEATURES_GIRO)
        y_clf = ds.fechou_venda.astype(int)
        y_reg = np.log1p(ds.tempo_patio_historico.astype(float))

        idx = np.arange(len(ds))
        if self.backend == "tabpfn" and len(idx) > TABPFN_MAX_AMOSTRAS:
            idx = np.random.default_rng(self.seed).choice(idx, TABPFN_MAX_AMOSTRAS, replace=False)
        train, test = train_test_split(
            idx, test_size=holdout, random_state=self.seed, stratify=y_clf[idx]
        )

        self._clf.fit(x_clf[train], y_clf[train])
        self._reg.fit(x_reg[train], y_reg[train])
        self._fitted = True

        auc = roc_auc_score(y_clf[test], self._clf.predict_proba(x_clf[test])[:, 1])
        mae = mean_absolute_error(
            ds.tempo_patio_historico[test], np.expm1(self._reg.predict(x_reg[test]))
        )
        patio = ds.tempo_patio_historico.astype(float)
        baseline = mean_absolute_error(patio[test], np.full(len(test), np.median(patio[train])))
        return {
            "auc_fechamento": float(auc),
            "mae_giro_dias": float(mae),
            "mae_giro_baseline_dias": float(baseline),
        }

    def predict(self, lead: LeadData, vehicle: VehicleAppraisalInput) -> PredictionOutput:
        if not self._fitted:
            raise RuntimeError("LDMPredictor.predict chamado antes de fit()")

        x_clf = np.array(
            [[
                vehicle.ano_modelo,
                vehicle.km_rodados,
                vehicle.valor_fipe,
                vehicle.percentual_fipe_pedido,
                lead.valor_entrada_pct,
            ]],
            dtype=float,
        )
        x_reg = x_clf[:, : len(FEATURES_GIRO)]

        km_ano = vehicle.km_rodados / max(1, ANO_REFERENCIA - vehicle.ano_modelo)
        mult_prob, mult_giro, ajustes = self.weights.multiplicadores(
            km_ano, lead.valor_entrada_pct
        )
        proba = float(self._clf.predict_proba(x_clf)[0, 1]) * mult_prob
        giro_base = float(np.expm1(self._reg.predict(x_reg)[0]))
        giro = max(1, int(round(giro_base * mult_giro)))
        teto_pct = avaliacao_fipe_recomendada(giro, self.margem_alvo, self.custos)

        return PredictionOutput(
            probabilidade_fechamento=round(min(max(proba, 0.0), 1.0) * 100.0, 1),
            giro_estimado_dias=giro,
            avaliacao_fipe_recomendada=teto_pct,
            valor_avaliacao_max=round(vehicle.valor_fipe * teto_pct / 100.0, 2),
            margem_liquida_alvo=self.margem_alvo,
            gap_pedido_pp=round(vehicle.percentual_fipe_pedido * 100.0 - teto_pct, 2),
            backend=self.backend,
            versao_pesos=self.weights.versao,
            ajustes_liquidez=ajustes,
        )

    def predict_raw(
        self, lead: Mapping[str, Any], vehicle: Mapping[str, Any]
    ) -> PredictionOutput:
        """Entrada vinda de dict (ex: State do grafo). FIPE nula -> fail-closed."""
        ensure_fipe(vehicle)
        return self.predict(
            LeadData.model_validate(lead), VehicleAppraisalInput.model_validate(vehicle)
        )


def ensure_fipe(vehicle: Mapping[str, Any]) -> float:
    """Levanta `FipeIndisponivelError` se a FIPE for ausente, nula, NaN ou <= 0."""
    fipe = vehicle.get("valor_fipe")
    if (
        fipe is None
        or isinstance(fipe, bool)
        or not isinstance(fipe, int | float)
        or not math.isfinite(fipe)
        or fipe <= 0
    ):
        raise FipeIndisponivelError(f"valor_fipe inválido ({fipe!r}); avaliação bloqueada.")
    return float(fipe)

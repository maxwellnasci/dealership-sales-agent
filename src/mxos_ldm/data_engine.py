"""Gerador de dataset sintético calibrado para o mercado de usados de Curitiba.

IMPORTANTE: os parâmetros abaixo são PREMISSAS de calibração (ordem de
grandeza plausível para Curitiba/PR em 2026), não dados reais de mercado. Eles
existem para treinar e testar o pipeline de ponta a ponta antes de plugar o
histórico real da revenda — quando ele chegar, substitui `generate_dataset`
sem mudar `model.py`.

Premissas:
  - Mix de segmentos com FIPE 0 km de referência (hatch popular a picape média);
    SUV compacto e picape giram mais rápido, sedan gira mais devagar.
  - Depreciação ~11%/ano composta, mais penalidade por km acima de 12,5 mil/ano.
  - Tempo de pátio cresce com idade, km e com o pedido acima da FIPE.
  - Fechamento cai forte quando o cliente pede acima da FIPE pelo carro da
    troca e sobe com a entrada.

Tudo é determinístico dado o `seed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mxos_ldm.schemas import FipeIndisponivelError

ANO_REFERENCIA = 2026
N_AMOSTRAS_PADRAO = 10_000

# (nome, FIPE 0 km de referência em R$, participação no mix, giro base em dias)
SEGMENTOS: tuple[tuple[str, float, float, float], ...] = (
    ("hatch_popular", 82_000.0, 0.30, 30.0),
    ("hatch_premium", 115_000.0, 0.12, 34.0),
    ("sedan_compacto", 108_000.0, 0.15, 40.0),
    ("sedan_medio", 165_000.0, 0.10, 44.0),
    ("suv_compacto", 145_000.0, 0.23, 26.0),
    ("picape_media", 240_000.0, 0.10, 33.0),
)

DEPRECIACAO_ANUAL = 0.11
KM_ANO_MEDIO = 12_500.0

# Colunas, na ordem em que entram em cada modelo.
FEATURES_FECHAMENTO: tuple[str, ...] = (
    "ano_modelo",
    "km_rodados",
    "valor_fipe",
    "percentual_fipe_pedido",
    "valor_entrada_pct",
)
FEATURES_GIRO: tuple[str, ...] = ("ano_modelo", "km_rodados", "valor_fipe")


@dataclass(frozen=True)
class SyntheticDataset:
    """Colunas do dataset como arrays numpy alinhados (uma linha = um negócio)."""

    ano_modelo: np.ndarray
    km_rodados: np.ndarray
    valor_fipe: np.ndarray
    percentual_fipe_pedido: np.ndarray
    valor_entrada_pct: np.ndarray
    tempo_patio_historico: np.ndarray
    fechou_venda: np.ndarray

    def __len__(self) -> int:
        return int(self.fechou_venda.shape[0])

    def matrix(self, columns: tuple[str, ...]) -> np.ndarray:
        return np.column_stack([getattr(self, c).astype(float) for c in columns])


def validate_dataset(ds: SyntheticDataset) -> None:
    """Fail-closed no treino: nenhuma linha com FIPE nula/NaN/não positiva."""
    fipe = ds.valor_fipe.astype(float)
    invalid = ~np.isfinite(fipe) | (fipe <= 0)
    if invalid.any():
        raise FipeIndisponivelError(
            f"{int(invalid.sum())} linha(s) com valor_fipe nulo/inválido; treino bloqueado."
        )


def generate_dataset(n: int = N_AMOSTRAS_PADRAO, seed: int = 42) -> SyntheticDataset:
    """Gera `n` negócios sintéticos de Curitiba (determinístico por `seed`)."""
    if n <= 0:
        raise ValueError("n deve ser positivo")
    rng = np.random.default_rng(seed)

    shares = np.array([s[2] for s in SEGMENTOS])
    seg = rng.choice(len(SEGMENTOS), size=n, p=shares / shares.sum())
    fipe_0km = np.array([s[1] for s in SEGMENTOS])[seg]
    giro_base = np.array([s[3] for s in SEGMENTOS])[seg]

    idade = rng.integers(1, 15, size=n)  # ano-modelo 2012..2025
    ano_modelo = ANO_REFERENCIA - idade

    km = idade * KM_ANO_MEDIO * rng.lognormal(mean=0.0, sigma=0.35, size=n)
    km = np.clip(np.round(km / 100.0) * 100.0, 500.0, 400_000.0).astype(int)

    excesso_km = np.clip((km - idade * KM_ANO_MEDIO) / 10_000.0, 0.0, None)
    fipe = fipe_0km * (1 - DEPRECIACAO_ANUAL) ** idade * (1 - 0.01 * excesso_km)
    fipe *= rng.normal(1.0, 0.06, size=n)
    fipe = np.round(np.clip(fipe, 12_000.0, None), 2)

    pct_pedido = np.round(np.clip(rng.normal(1.0, 0.07, size=n), 0.80, 1.25), 3)
    entrada = np.round(np.clip(rng.beta(2.0, 5.0, size=n), 0.0, 0.9), 3)

    patio_medio = (
        giro_base
        + 2.2 * idade
        + 1.4 * km / 10_000.0
        + 60.0 * np.clip(pct_pedido - 1.0, 0.0, None)
        + 8.0 * (fipe > 150_000.0)
    )
    # Gamma com média `patio_medio` e cauda longa (carros "encalhados").
    patio = rng.gamma(shape=4.0, scale=patio_medio / 4.0)
    patio = np.clip(np.round(patio), 3, 240).astype(int)

    logit = (
        0.6
        - 12.0 * (pct_pedido - 0.97)
        + 4.0 * (entrada - 0.25)
        - 0.06 * (idade - 6)
        - 0.25 * np.log(fipe / 60_000.0)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    fechou = (rng.random(n) < p).astype(int)

    return SyntheticDataset(
        ano_modelo=ano_modelo.astype(int),
        km_rodados=km,
        valor_fipe=fipe,
        percentual_fipe_pedido=pct_pedido,
        valor_entrada_pct=entrada,
        tempo_patio_historico=patio,
        fechou_venda=fechou,
    )

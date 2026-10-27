"""Testes do mxos_ldm: schemas, dataset sintético, matemática de margem,
pipeline preditivo (xgboost/sklearn/tabpfn) e Card Executivo do vendedor.

Fail-closed é o eixo: FIPE nula/NaN/zero bloqueia validação, treino,
previsão e o card — nunca sai um teto de avaliação inventado.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import types
from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from dealership_agent.schemas.lead import LeadProfile
from mxos_ldm import model as ldm_model
from mxos_ldm.data_engine import (
    ANO_REFERENCIA,
    FEATURES_FECHAMENTO,
    FEATURES_GIRO,
    N_AMOSTRAS_PADRAO,
    generate_dataset,
    validate_dataset,
)
from mxos_ldm.model import (
    MARGEM_LIQUIDA_ALVO,
    CustosRevenda,
    LDMPredictor,
    avaliacao_fipe_recomendada,
    ensure_fipe,
    margem_liquida,
)
from mxos_ldm.sales_bridge import (
    TZ_CURITIBA,
    build_seller_card,
    classificar_prioridade,
    formatar_numero,
    formatar_pct,
    formatar_reais,
    proximo_envio,
)
from mxos_ldm.schemas import (
    FipeIndisponivelError,
    LeadData,
    PredictionOutput,
    VehicleAppraisalInput,
)
from mxos_ldm.weights import credit_approval, faixa_entrada, km_liquidity_factor, load_weights

# ---------------------------------------------------------------- fixtures


def _vehicle(**overrides) -> dict:
    base = {
        "ano_modelo": 2019,
        "km_rodados": 78_000,
        "valor_fipe": 62_500.0,
        "percentual_fipe_pedido": 0.95,
        "descricao": "VW Gol 1.6 2019",
    }
    base.update(overrides)
    return base


def _lead(**overrides) -> dict:
    base = {
        "lead_id": "5541999990000",
        "nome": "Ana",
        "valor_entrada_pct": 0.30,
        "veiculo_interesse": "Jeep Renegade 2022",
    }
    base.update(overrides)
    return base


def _prediction(**overrides) -> PredictionOutput:
    base = {
        "probabilidade_fechamento": 72.4,
        "giro_estimado_dias": 30,
        "avaliacao_fipe_recomendada": 82.98,
        "valor_avaliacao_max": 51_862.5,
        "margem_liquida_alvo": 0.12,
        "gap_pedido_pp": 12.02,
        "backend": "xgboost",
    }
    base.update(overrides)
    return PredictionOutput(**base)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset()


@pytest.fixture(scope="module")
def trained(dataset):
    predictor = LDMPredictor(backend="xgboost")
    metrics = predictor.fit(dataset)
    return predictor, metrics


NOW = datetime(2026, 10, 26, 7, 0, tzinfo=TZ_CURITIBA)

# ---------------------------------------------------------------- schemas


def test_schemas_aceitam_entrada_valida():
    v = VehicleAppraisalInput(**_vehicle())
    lead = LeadData(**_lead())
    assert v.valor_fipe == 62_500.0
    assert lead.valor_entrada_pct == 0.30
    # strict aceita int para float (sem perda), mas não string.
    assert VehicleAppraisalInput(**_vehicle(valor_fipe=62_500)).valor_fipe == 62_500


@pytest.mark.parametrize("fipe", [None, float("nan"), float("inf"), 0.0, -1.0, "62500"])
def test_vehicle_rejeita_fipe_invalida(fipe):
    with pytest.raises(ValidationError) as exc:
        VehicleAppraisalInput(**_vehicle(valor_fipe=fipe))
    assert exc.value.errors()[0]["loc"] == ("valor_fipe",)


def test_vehicle_rejeita_fipe_ausente():
    data = _vehicle()
    del data["valor_fipe"]
    with pytest.raises(ValidationError):
        VehicleAppraisalInput(**data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ano_modelo", 1980),
        ("ano_modelo", ANO_REFERENCIA + 5),
        ("ano_modelo", "2019"),
        ("km_rodados", -1),
        ("km_rodados", 78_000.5),
        ("percentual_fipe_pedido", 0.0),
        ("percentual_fipe_pedido", 2.5),
    ],
)
def test_vehicle_rejeita_campos_invalidos(field, value):
    with pytest.raises(ValidationError):
        VehicleAppraisalInput(**_vehicle(**{field: value}))


def test_schemas_proibem_campos_extras_e_sao_imutaveis():
    with pytest.raises(ValidationError):
        VehicleAppraisalInput(**_vehicle(cor="prata"))
    v = VehicleAppraisalInput(**_vehicle())
    with pytest.raises(ValidationError):
        v.valor_fipe = 1.0


@pytest.mark.parametrize(
    "overrides", [{"valor_entrada_pct": 1.1}, {"valor_entrada_pct": -0.1}, {"lead_id": ""}]
)
def test_lead_rejeita_invalidos(overrides):
    with pytest.raises(ValidationError):
        LeadData(**_lead(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"probabilidade_fechamento": 100.1},
        {"probabilidade_fechamento": -0.1},
        {"giro_estimado_dias": 0},
        {"avaliacao_fipe_recomendada": 101.0},
        {"margem_liquida_alvo": 1.0},
        {"valor_avaliacao_max": -1.0},
    ],
)
def test_prediction_output_limites(overrides):
    with pytest.raises(ValidationError):
        _prediction(**overrides)


# ---------------------------------------------------------------- data engine


def test_dataset_tamanho_e_faixas(dataset):
    assert len(dataset) == N_AMOSTRAS_PADRAO == 10_000
    assert dataset.ano_modelo.min() >= ANO_REFERENCIA - 14
    assert dataset.ano_modelo.max() <= ANO_REFERENCIA - 1
    assert dataset.km_rodados.min() >= 500
    assert (dataset.valor_fipe >= 12_000).all()
    assert dataset.percentual_fipe_pedido.min() >= 0.80
    assert dataset.percentual_fipe_pedido.max() <= 1.25
    assert dataset.valor_entrada_pct.min() >= 0.0
    assert dataset.valor_entrada_pct.max() <= 0.9
    assert dataset.tempo_patio_historico.min() >= 3
    assert dataset.tempo_patio_historico.max() <= 240
    assert set(np.unique(dataset.fechou_venda)) == {0, 1}
    assert 0.2 < dataset.fechou_venda.mean() < 0.8


def test_dataset_determinismo_por_seed():
    a, b, c = generate_dataset(500, seed=7), generate_dataset(500, seed=7), generate_dataset(500)
    np.testing.assert_array_equal(a.valor_fipe, b.valor_fipe)
    np.testing.assert_array_equal(a.fechou_venda, b.fechou_venda)
    assert not np.array_equal(a.valor_fipe, c.valor_fipe)


def test_dataset_calibracao_direcional(dataset):
    caro = dataset.percentual_fipe_pedido > 1.05
    barato = dataset.percentual_fipe_pedido < 0.95
    assert dataset.fechou_venda[caro].mean() < dataset.fechou_venda[barato].mean()

    velho = dataset.ano_modelo <= ANO_REFERENCIA - 10
    novo = dataset.ano_modelo >= ANO_REFERENCIA - 3
    assert dataset.tempo_patio_historico[velho].mean() > dataset.tempo_patio_historico[novo].mean()
    assert dataset.valor_fipe[velho].mean() < dataset.valor_fipe[novo].mean()


def test_dataset_matrix(dataset):
    assert dataset.matrix(FEATURES_FECHAMENTO).shape == (10_000, 5)
    assert dataset.matrix(FEATURES_GIRO).dtype == float


def test_dataset_n_invalido():
    with pytest.raises(ValueError):
        generate_dataset(0)


@pytest.mark.parametrize("bad", [np.nan, None, 0.0, -5.0])
def test_validate_dataset_fail_closed(bad):
    ds = generate_dataset(50)
    fipe = ds.valor_fipe.astype(object)
    fipe[3] = bad
    ds_bad = dataclasses.replace(ds, valor_fipe=fipe)
    with pytest.raises(FipeIndisponivelError, match="1 linha"):
        validate_dataset(ds_bad)


# ---------------------------------------------------------------- matemática de margem


def test_avaliacao_valores_exatos():
    # giro 30: revenda = 1 - 0.0005*30 = 0.985 ; custos = 0.025 + 0.0004*30 = 0.037
    # teto = 0.985*0.88 - 0.037 = 0.8298
    assert avaliacao_fipe_recomendada(30) == 82.98
    # giro 0: 1*0.88 - 0.025 = 0.855
    assert avaliacao_fipe_recomendada(0) == 85.5


@pytest.mark.parametrize("giro", [0, 7, 30, 45, 90, 180])
def test_avaliacao_garante_margem_alvo(giro):
    teto = avaliacao_fipe_recomendada(giro)
    assert margem_liquida(teto / 100.0, giro) == pytest.approx(MARGEM_LIQUIDA_ALVO, abs=1e-4)


def test_avaliacao_monotonica_no_giro_e_na_margem():
    tetos = [avaliacao_fipe_recomendada(g) for g in range(0, 200, 10)]
    assert tetos == sorted(tetos, reverse=True)
    assert avaliacao_fipe_recomendada(30, margem_alvo=0.20) < avaliacao_fipe_recomendada(30)


def test_avaliacao_truncada():
    assert avaliacao_fipe_recomendada(2_000) == 0.0
    generosa = CustosRevenda(preco_revenda_pct_fipe=2.0, custo_preparacao_pct=0.0)
    assert avaliacao_fipe_recomendada(0, custos=generosa) == 100.0


@pytest.mark.parametrize(
    "giro,margem", [(-1, 0.12), (math.nan, 0.12), (math.inf, 0.12), (30, 0.0), (30, 1.0)]
)
def test_avaliacao_entradas_invalidas(giro, margem):
    with pytest.raises(ValueError):
        avaliacao_fipe_recomendada(giro, margem_alvo=margem)


def test_margem_liquida_formula():
    # compra a 80% FIPE, gira em 0 dias: (1 - 0.80 - 0.025) / 1 = 0.175
    assert margem_liquida(0.80, 0) == pytest.approx(0.175)


# ---------------------------------------------------------------- ensure_fipe


@pytest.mark.parametrize("fipe", [None, float("nan"), float("-inf"), 0, -1.0, "62500", True])
def test_ensure_fipe_fail_closed(fipe):
    with pytest.raises(FipeIndisponivelError):
        ensure_fipe({"valor_fipe": fipe})


def test_ensure_fipe_ausente_e_valida():
    with pytest.raises(FipeIndisponivelError):
        ensure_fipe({})
    assert ensure_fipe({"valor_fipe": 50_000}) == 50_000.0


# ---------------------------------------------------------------- modelo


def test_predict_antes_do_fit():
    with pytest.raises(RuntimeError):
        LDMPredictor(backend="sklearn").predict(
            LeadData(**_lead()), VehicleAppraisalInput(**_vehicle())
        )


def test_xgboost_treina_com_qualidade(trained):
    predictor, metrics = trained
    assert predictor.backend == "xgboost"
    assert metrics["auc_fechamento"] > 0.70
    # ruído gamma (CV 50%) domina o giro sintético: exigimos bater o baseline da mediana
    assert metrics["mae_giro_dias"] < 0.95 * metrics["mae_giro_baseline_dias"]


def test_predict_saida_coerente(trained):
    predictor, _ = trained
    vehicle = VehicleAppraisalInput(**_vehicle())
    out = predictor.predict(LeadData(**_lead()), vehicle)

    assert 0.0 <= out.probabilidade_fechamento <= 100.0
    assert out.giro_estimado_dias >= 1
    assert out.avaliacao_fipe_recomendada == avaliacao_fipe_recomendada(out.giro_estimado_dias)
    assert out.valor_avaliacao_max == pytest.approx(
        vehicle.valor_fipe * out.avaliacao_fipe_recomendada / 100.0, abs=0.01
    )
    assert out.gap_pedido_pp == pytest.approx(95.0 - out.avaliacao_fipe_recomendada, abs=0.01)
    assert out.margem_liquida_alvo == MARGEM_LIQUIDA_ALVO
    assert out.backend == "xgboost"


def test_predict_direcional(trained):
    predictor, _ = trained
    lead = LeadData(**_lead())
    barato = predictor.predict(lead, VehicleAppraisalInput(**_vehicle(percentual_fipe_pedido=0.88)))
    caro = predictor.predict(lead, VehicleAppraisalInput(**_vehicle(percentual_fipe_pedido=1.20)))
    assert caro.probabilidade_fechamento < barato.probabilidade_fechamento

    novo_v = VehicleAppraisalInput(**_vehicle(ano_modelo=2024, km_rodados=15_000))
    velho_v = VehicleAppraisalInput(**_vehicle(ano_modelo=2012, km_rodados=220_000))
    novo, velho = predictor.predict(lead, novo_v), predictor.predict(lead, velho_v)
    assert velho.giro_estimado_dias > novo.giro_estimado_dias
    assert velho.avaliacao_fipe_recomendada < novo.avaliacao_fipe_recomendada


def test_predict_raw_ok_e_fail_closed(trained):
    predictor, _ = trained
    assert predictor.predict_raw(_lead(), _vehicle()).backend == "xgboost"
    for bad in (None, float("nan"), 0.0):
        with pytest.raises(FipeIndisponivelError):
            predictor.predict_raw(_lead(), _vehicle(valor_fipe=bad))


def test_sklearn_backend():
    predictor = LDMPredictor(backend="sklearn")
    metrics = predictor.fit(generate_dataset(3_000, seed=1))
    assert predictor.backend == "sklearn"
    assert metrics["auc_fechamento"] > 0.65
    assert predictor.predict_raw(_lead(), _vehicle()).backend == "sklearn"


def test_auto_backend(monkeypatch):
    assert LDMPredictor(backend="auto").backend == "xgboost"
    monkeypatch.setattr(ldm_model, "_xgboost_available", lambda: False)
    assert LDMPredictor(backend="auto").backend == "sklearn"


def test_backend_desconhecido():
    with pytest.raises(ValueError, match="backend desconhecido"):
        LDMPredictor(backend="lightgbm")  # type: ignore[arg-type]


def test_tabpfn_ausente(monkeypatch):
    monkeypatch.setitem(sys.modules, "tabpfn", None)
    with pytest.raises(ImportError, match="tabpfn"):
        LDMPredictor(backend="tabpfn")


def test_tabpfn_backend_com_subamostragem(monkeypatch):
    fitted_sizes: list[int] = []

    class FakeClf(HistGradientBoostingClassifier):
        def fit(self, x, y):
            fitted_sizes.append(len(x))
            return super().fit(x, y)

    fake = types.ModuleType("tabpfn")
    fake.TabPFNClassifier = FakeClf
    fake.TabPFNRegressor = HistGradientBoostingRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake)

    predictor = LDMPredictor(backend="tabpfn")
    predictor.fit(generate_dataset(ldm_model.TABPFN_MAX_AMOSTRAS + 500), holdout=0.2)
    assert predictor.backend == "tabpfn"
    # treino = 80% do teto de amostras do TabPFN, não do dataset inteiro
    assert fitted_sizes == [int(ldm_model.TABPFN_MAX_AMOSTRAS * 0.8)]
    assert predictor.predict_raw(_lead(), _vehicle()).backend == "tabpfn"


def test_fit_fail_closed_com_fipe_nula():
    ds = generate_dataset(200)
    fipe = ds.valor_fipe.copy()
    fipe[10] = np.nan
    with pytest.raises(FipeIndisponivelError):
        LDMPredictor(backend="sklearn").fit(dataclasses.replace(ds, valor_fipe=fipe))


def test_predict_trunca_saidas_extremas():
    class Clf:
        def predict_proba(self, x):
            return np.array([[-0.2, 1.2]])

    class Reg:
        def predict(self, x):
            return np.array([-50.0])  # expm1 -> ~-1 dia

    predictor = LDMPredictor(backend="sklearn")
    predictor._clf, predictor._reg, predictor._fitted = Clf(), Reg(), True
    out = predictor.predict(LeadData(**_lead()), VehicleAppraisalInput(**_vehicle()))
    assert out.probabilidade_fechamento == 100.0
    assert out.giro_estimado_dias == 1


# ---------------------------------------------------------------- sales bridge


@pytest.mark.parametrize(
    "now,esperado",
    [
        (datetime(2026, 10, 26, 7, 0, tzinfo=TZ_CURITIBA), datetime(2026, 10, 26, 8, 30)),
        (datetime(2026, 10, 26, 8, 30, tzinfo=TZ_CURITIBA), datetime(2026, 10, 27, 8, 30)),
        (datetime(2026, 10, 26, 18, 0, tzinfo=TZ_CURITIBA), datetime(2026, 10, 27, 8, 30)),
        (datetime(2026, 10, 26, 6, 0), datetime(2026, 10, 26, 8, 30)),  # naive = local
        (datetime(2026, 10, 26, 12, 0, tzinfo=UTC), datetime(2026, 10, 27, 8, 30)),  # 09h local
        (datetime(2026, 12, 31, 23, 0, tzinfo=TZ_CURITIBA), datetime(2027, 1, 1, 8, 30)),
    ],
)
def test_proximo_envio(now, esperado):
    assert proximo_envio(now) == esperado.replace(tzinfo=TZ_CURITIBA)


def test_formatar_reais():
    assert formatar_reais(1_234_567.8) == "R$ 1.234.567,80"
    assert formatar_reais(0) == "R$ 0,00"


@pytest.mark.parametrize(
    "prob,esperado", [(60.0, "alta"), (59.9, "media"), (35.0, "media"), (34.9, "baixa")]
)
def test_classificar_prioridade(prob, esperado):
    assert classificar_prioridade(prob) == esperado


def _state(**overrides) -> dict:
    state = {
        "lead_profile": LeadProfile(name="Ana", phone="5541999990000", trade_in=True),
        "ldm_lead": LeadData(**_lead()),
        "ldm_vehicle": VehicleAppraisalInput(**_vehicle()),
    }
    state.update(overrides)
    return state


def test_card_com_previsao_pronta():
    card = build_seller_card(_state(ldm_prediction=_prediction()), now=NOW)
    assert not card.bloqueado
    assert card.prioridade == "alta"
    assert card.envio_agendado_para == datetime(2026, 10, 26, 8, 30, tzinfo=TZ_CURITIBA)
    assert card.texto.splitlines() == [
        "📋 CARD EXECUTIVO — 26/10 08:30",
        "Lead: Ana (5541999990000)",
        "Interesse: Jeep Renegade 2022",
        "Prioridade: ALTA — 72,4% de fechamento",
        "Entrada: 30% do valor",
        "Troca: VW Gol 1.6 2019 · 78.000 km · FIPE R$ 62.500,00",
        "Giro estimado da troca: 30 dias",
        "Avaliação máxima: R$ 51.862,50 (83,0% FIPE, margem líquida 12%)",
        "Cliente pede 95,0% da FIPE; teto é 83,0% → negociar 12,0 p.p. para baixo.",
    ]


def test_card_pedido_dentro_do_teto_e_campos_opcionais():
    state = _state(
        ldm_lead=_lead(veiculo_interesse=None),
        ldm_vehicle=_vehicle(descricao=None, percentual_fipe_pedido=0.80),
        ldm_prediction=_prediction(probabilidade_fechamento=20.0, gap_pedido_pp=-2.98).model_dump(),
    )
    card = build_seller_card(state, now=NOW)
    assert card.prioridade == "baixa"
    assert "Interesse:" not in card.texto
    assert "Troca: ano 2019" in card.texto
    assert "dentro do teto (83,0%) — pode fechar" in card.texto


def test_card_calcula_com_predictor(trained):
    predictor, _ = trained
    card = build_seller_card(_state(), predictor=predictor)
    assert not card.bloqueado
    assert card.prioridade in {"alta", "media", "baixa"}
    assert card.envio_agendado_para > datetime.now(TZ_CURITIBA)
    assert "Avaliação máxima: R$" in card.texto


@pytest.mark.parametrize("fipe", [None, float("nan"), 0.0])
def test_card_fail_closed_fipe(trained, fipe):
    predictor, _ = trained
    card = build_seller_card(
        _state(ldm_vehicle=_vehicle(valor_fipe=fipe), ldm_prediction=_prediction()),
        predictor=predictor,
        now=NOW,
    )
    assert card.bloqueado
    assert card.prioridade == "bloqueado"
    assert card.motivo_bloqueio == "FIPE indisponível — avaliação não calculada"
    assert "R$" not in card.texto  # nenhum número de avaliação vaza


def test_card_bloqueios_diversos():
    sem_troca = build_seller_card(_state(ldm_vehicle=None), now=NOW)
    assert sem_troca.motivo_bloqueio == "sem dados do carro da troca"

    km_ruim = build_seller_card(_state(ldm_vehicle=_vehicle(km_rodados=-5)), now=NOW)
    assert km_ruim.motivo_bloqueio == "dados inválidos (km_rodados)"

    sem_previsao = build_seller_card(_state(), now=NOW)
    assert sem_previsao.motivo_bloqueio == "previsão do LDM indisponível"


def test_card_fallback_identidade_do_lead_profile():
    card = build_seller_card(_state(ldm_lead=None), now=NOW)
    # sem ldm_lead, LeadData não valida -> bloqueado, mas identificado pelo profile
    assert card.motivo_bloqueio == "dados inválidos (lead_id, valor_entrada_pct)"
    assert card.lead_id == "5541999990000"
    assert "Ana" in card.texto

    anonimo = build_seller_card({"ldm_vehicle": None}, now=NOW)
    assert anonimo.lead_id == "desconhecido"
    assert "Lead (desconhecido)" in anonimo.texto


# ---------------------------------------------------------------- pesos de Curitiba (Antigravity)

PESOS_PATH = "data/mxos_ldm_weights.json"


@pytest.fixture(scope="module")
def pesos():
    return load_weights(PESOS_PATH)


def _pesos_raw() -> dict:
    with open(PESOS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_formatar_pct_virgula_brasileira():
    assert formatar_pct(76.6) == "76,6%"
    assert formatar_pct(30.0000001, 0) == "30%"
    assert formatar_numero(12.04) == "12,0"


def test_load_weights_arquivo_real(pesos):
    assert pesos.versao == "mxos_ldm-1.0.0"
    assert pesos.metadata.benchmark_km_per_year == 15_000.0
    assert len(pesos.models_curitiba_top10) == 10
    assert set(pesos.curitiba_hubs) == {"linha_verde", "mario_tourinho"}


def test_load_weights_ausente_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="pesos do mxos_ldm"):
        load_weights(tmp_path / "nao_existe.json")


@pytest.mark.parametrize(
    "mutacao",
    [
        lambda d: d.update(campo_desconhecido=1),
        lambda d: d["metadata"].update(engine="outro"),
        lambda d: d["metadata"].update(app_usage_threshold_km_per_year=10_000.0),
        lambda d: d["credit_scoring_weights"]["down_payment_spline"].update(low_ltv_cutoff=0.5),
        lambda d: d["models_curitiba_top10"]["Onix"].update(km_penalty_gamma=-0.1),
        lambda d: d["models_curitiba_top10"]["Onix"].update(base_days_on_lot_D0="18"),
        lambda d: d["curitiba_hubs"].update(batel={}),
        lambda d: d.update(models_curitiba_top10={}),
    ],
)
def test_load_weights_malformado_fail_closed(tmp_path, mutacao):
    data = _pesos_raw()
    mutacao(data)
    path = tmp_path / "w.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        load_weights(path)


@pytest.mark.parametrize(
    "total_km,idade,esperado",
    [
        (24_000, 2.0, (12_000.0, 1.008)),   # abaixo do benchmark: bônus
        (0, 2.0, (0.0, 1.04)),              # bônus máximo de +4%
        (30_000, 2.0, (15_000.0, 1.0)),     # exatamente no benchmark
        (48_000, 2.0, (24_000.0, 0.749)),   # acima: decaimento γ (ex-app em formação)
        (60_000, 2.0, (30_000.0, 0.552)),   # acima de 25 mil: penhasco η
        (400_000, 1.0, (400_000.0, 0.15)),  # piso
        (6_000, 0, (12_000.0, 1.008)),      # idade 0 -> 0,5 ano
    ],
)
def test_km_liquidity_factor(pesos, total_km, idade, esperado):
    assert km_liquidity_factor(pesos, "Onix", total_km, idade) == esperado


def test_km_liquidity_factor_monotonico(pesos):
    phis = [km_liquidity_factor(pesos, "HB20", km, 3.0)[1] for km in range(45_000, 120_000, 5_000)]
    assert phis == sorted(phis, reverse=True)


@pytest.mark.parametrize(
    "entrada,esperado",
    [
        (0.15, (0.361, 0.0200)),
        (0.30, (0.532, 0.0189)),
        (0.45, (0.726, 0.0182)),
        (1.50, (0.996, 0.0161)),   # entrada truncada em 100%
        (-0.1, (0.199, 0.0233)),   # entrada truncada em 0%
    ],
)
def test_credit_approval_spline(pesos, entrada, esperado):
    assert credit_approval(pesos, entrada) == esperado


def test_credit_approval_score_e_dti(pesos):
    base, _ = credit_approval(pesos, 0.30)
    assert credit_approval(pesos, 0.30, credit_score=800)[0] > base
    assert credit_approval(pesos, 0.30, dti_ratio=0.45)[0] < base
    assert credit_approval(pesos, 0.30, vehicle_age_years=8)[0] < base
    assert credit_approval(pesos, 0.30, credit_score=2_000)[1] == 0.0129  # piso da taxa


@pytest.mark.parametrize(
    "entrada,rotulo", [(0.10, "entrada<20%"), (0.20, "entrada 20%-40%"), (0.41, "entrada>40%")]
)
def test_faixa_entrada(pesos, entrada, rotulo):
    assert faixa_entrada(pesos, entrada) == rotulo


@pytest.fixture(scope="module")
def sklearn_predictor(dataset):
    p = LDMPredictor(backend="sklearn")
    p.fit(dataset)
    return p


def test_predictor_aplica_pesos_curitiba(sklearn_predictor, pesos, monkeypatch):
    lead = LeadData(**_lead(valor_entrada_pct=0.15))
    # Onix 2024 com 48 mil km -> 24 mil km/ano -> φ = 0,749
    vehicle = VehicleAppraisalInput(**_vehicle(ano_modelo=2024, km_rodados=48_000, modelo="Onix"))
    base = sklearn_predictor.predict(lead, vehicle)
    monkeypatch.setattr(sklearn_predictor, "weights", pesos)
    ajustado = sklearn_predictor.predict(lead, vehicle)

    assert base.versao_pesos == "neutro"
    assert base.km_liquidity_factor is None and base.credit_approval_prob is None
    assert base.ajustes_liquidez == []

    assert ajustado.versao_pesos == "mxos_ldm-1.0.0"
    assert ajustado.km_liquidity_factor == 0.749
    assert ajustado.credit_approval_prob == 0.361
    assert ajustado.ajustes_liquidez == ["km/ano 24.000 φ=0.749", "entrada<20%"]
    # giro = giro_base / φ ; teto sai do giro ajustado
    assert ajustado.giro_estimado_dias > base.giro_estimado_dias
    assert ajustado.avaliacao_fipe_recomendada == avaliacao_fipe_recomendada(
        ajustado.giro_estimado_dias
    )
    # entrada já é feature do classificador: probabilidade NÃO é multiplicada
    assert ajustado.probabilidade_fechamento == base.probabilidade_fechamento


@pytest.mark.parametrize(
    "modelo,rotulos",
    [(None, ["entrada 20%-40%"]), ("Gol", ["Gol fora do top 10 (sem φ de km)", "entrada 20%-40%"])],
)
def test_predictor_modelo_fora_do_top10(sklearn_predictor, pesos, monkeypatch, modelo, rotulos):
    vehicle = VehicleAppraisalInput(**_vehicle(modelo=modelo))
    base = sklearn_predictor.predict(LeadData(**_lead()), vehicle)
    monkeypatch.setattr(sklearn_predictor, "weights", pesos)
    out = sklearn_predictor.predict(LeadData(**_lead()), vehicle)
    assert out.km_liquidity_factor is None
    assert out.giro_estimado_dias == base.giro_estimado_dias
    assert out.ajustes_liquidez == rotulos


def test_card_mostra_credito_e_ajustes():
    pred = _prediction(
        versao_pesos="mxos_ldm-1.0.0",
        ajustes_liquidez=["km/ano 24.000 φ=0.749", "entrada<20%"],
        credit_approval_prob=0.361,
        km_liquidity_factor=0.749,
    )
    linhas = build_seller_card(_state(ldm_prediction=pred), now=NOW).texto.splitlines()
    assert linhas[-2:] == [
        "Aprovação de crédito estimada: 36,1%",
        "Ajustes de liquidez (mxos_ldm-1.0.0): km/ano 24.000 φ=0.749, entrada<20%",
    ]

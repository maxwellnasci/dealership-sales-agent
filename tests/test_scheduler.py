"""Agendador do Card Executivo: fuso de São Paulo, entrega pontual às 08h30,
retry exponencial, recuperação após restart e integração com grafo/API.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from tenacity import wait_none

from dealership_agent import main
from dealership_agent.config import settings
from dealership_agent.graph.builder import build_graph
from dealership_agent.graph.state import initial_state
from dealership_agent.llm import TemplateLLMClient
from dealership_agent.persistence.checkpointer import checkpoint_serde
from dealership_agent.scheduler import (
    MAX_ATTEMPTS,
    CardDispatcher,
    CardStatus,
    CardStore,
    NotifierError,
    RecordingNotifier,
    WebhookNotifier,
)
from dealership_agent.scheduler.store import QueuedCard
from mxos_ldm.data_engine import generate_dataset
from mxos_ldm.model import LDMPredictor
from mxos_ldm.sales_bridge import TZ_CURITIBA, SellerCard, proximo_envio
from mxos_ldm.schemas import VehicleAppraisalInput

SP = TZ_CURITIBA  # America/Sao_Paulo
TRADE_IN = {
    "ano_modelo": 2019,
    "km_rodados": 78_000,
    "valor_fipe": 62_500.0,
    "percentual_fipe_pedido": 0.95,
}


def sp(*args) -> datetime:
    return datetime(*args, tzinfo=SP)


def make_card(lead_id: str = "+5541900000001", gerado_em: datetime | None = None, texto="card"):
    gerado_em = gerado_em or sp(2026, 10, 26, 23, 10)
    return SellerCard(
        lead_id=lead_id,
        bloqueado=False,
        motivo_bloqueio=None,
        prioridade="alta",
        texto=texto,
        envio_agendado_para=proximo_envio(gerado_em),
    )


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class FlakyNotifier(RecordingNotifier):
    def __init__(self, failures: int, exc: Exception | None = None) -> None:
        super().__init__()
        self.failures = failures
        self.exc = exc or NotifierError("webhook fora do ar")
        self.calls = 0

    async def send(self, card: QueuedCard) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        await super().send(card)


@pytest.fixture
async def store(tmp_path):
    s = await CardStore.open(tmp_path / "scheduler.sqlite")
    yield s
    await s.close()


def dispatcher(store, notifier, clock) -> CardDispatcher:
    return CardDispatcher(store, notifier, clock=clock, retry_wait=wait_none())


# ---------------------------------------------------------------- fuso de São Paulo


async def test_card_da_madrugada_agenda_para_0830_sp(store):
    await store.enqueue(make_card(gerado_em=sp(2026, 10, 26, 23, 10)))
    (card,) = await store.list()
    assert card.status is CardStatus.PENDING
    # 08h30 de São Paulo (UTC-3, sem horário de verão) = 11h30 UTC
    assert card.deliver_at == datetime(2026, 10, 27, 11, 30, tzinfo=UTC)
    assert card.deliver_at.astimezone(SP) == sp(2026, 10, 27, 8, 30)


async def test_card_gerado_antes_das_0830_sai_no_mesmo_dia(store):
    await store.enqueue(make_card(gerado_em=sp(2026, 10, 27, 2, 0)))
    (card,) = await store.list()
    assert card.deliver_at.astimezone(SP) == sp(2026, 10, 27, 8, 30)


async def test_horario_sem_fuso_e_rejeitado(store):
    with pytest.raises(ValueError, match="fuso"):
        await store.enqueue(make_card(), now=datetime(2026, 10, 26, 23, 0))


# ---------------------------------------------------------------- envio pontual


async def test_envio_pontual_as_0830(store):
    await store.enqueue(make_card())
    clock = Clock(sp(2026, 10, 27, 8, 29, 59))
    notifier = RecordingNotifier()
    d = dispatcher(store, notifier, clock)

    assert await d.run_once() == 0  # 08:29:59 — ainda não
    assert notifier.sent == []

    clock.now = sp(2026, 10, 27, 8, 30)
    assert await d.run_once() == 1
    (card,) = await store.list(CardStatus.SENT)
    assert card.sent_at == clock.now  # timestamp de entrega registrado
    assert card.attempts == 1
    assert len(notifier.sent) == 1

    clock.now = sp(2026, 10, 27, 8, 31)
    assert await d.run_once() == 0  # não reenvia


async def test_servidor_fora_do_ar_as_0830_entrega_atrasado(store):
    await store.enqueue(make_card())
    notifier = RecordingNotifier()
    await dispatcher(store, notifier, Clock(sp(2026, 10, 27, 10, 15))).run_once()
    assert len(notifier.sent) == 1


async def test_card_de_amanha_nao_sai_hoje(store):
    await store.enqueue(make_card(gerado_em=sp(2026, 10, 27, 9, 0)))  # depois das 08h30
    clock = Clock(sp(2026, 10, 27, 23, 59))
    d = dispatcher(store, RecordingNotifier(), clock)
    assert await d.run_once() == 0
    clock.now = sp(2026, 10, 28, 8, 30)
    assert await d.run_once() == 1


# ---------------------------------------------------------------- idempotência


async def test_enqueue_idempotente_mantem_texto_mais_recente(store):
    await store.enqueue(make_card(texto="v1"))
    await store.enqueue(make_card(texto="v2", gerado_em=sp(2026, 10, 27, 1, 0)))
    (card,) = await store.list()
    assert card.texto == "v2"


async def test_card_enviado_nao_e_reaberto(store):
    await store.enqueue(make_card(texto="v1"))
    await dispatcher(store, RecordingNotifier(), Clock(sp(2026, 10, 27, 8, 30))).run_once()
    await store.enqueue(make_card(texto="v2"))
    (card,) = await store.list()
    assert card.status is CardStatus.SENT
    assert card.texto == "v1"


# ---------------------------------------------------------------- retry exponencial


async def test_retry_recupera_falha_transitoria(store):
    await store.enqueue(make_card())
    notifier = FlakyNotifier(failures=2)
    await dispatcher(store, notifier, Clock(sp(2026, 10, 27, 8, 30))).run_once()
    (card,) = await store.list()
    assert card.status is CardStatus.SENT
    assert card.attempts == 3
    assert notifier.calls == 3


async def test_retry_esgotado_marca_failed(store, caplog):
    await store.enqueue(make_card())
    notifier = FlakyNotifier(failures=99)
    with caplog.at_level(logging.ERROR):
        await dispatcher(store, notifier, Clock(sp(2026, 10, 27, 8, 30))).run_once()
    (card,) = await store.list()
    assert card.status is CardStatus.FAILED
    assert card.attempts == MAX_ATTEMPTS == 3
    assert card.last_error == "webhook fora do ar"
    assert card.sent_at is None
    assert "FAILED após 3 tentativas" in caplog.text


async def test_backoff_exponencial_padrao(store):
    await store.enqueue(make_card())
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    d = CardDispatcher(
        store, FlakyNotifier(failures=99), clock=Clock(sp(2026, 10, 27, 8, 30)), sleep=fake_sleep
    )
    await d.run_once()
    assert sleeps == [2, 4]  # 3 tentativas -> 2 esperas, dobrando


async def test_erro_inesperado_falha_sem_retry(store):
    await store.enqueue(make_card())
    notifier = FlakyNotifier(failures=99, exc=ValueError("payload inválido"))
    await dispatcher(store, notifier, Clock(sp(2026, 10, 27, 8, 30))).run_once()
    (card,) = await store.list()
    assert card.status is CardStatus.FAILED
    assert card.attempts == 1
    assert card.last_error == "ValueError: payload inválido"


# ---------------------------------------------------------------- recuperação após restart


async def test_pendentes_sobrevivem_ao_restart(tmp_path):
    path = tmp_path / "scheduler.sqlite"
    first = await CardStore.open(path)
    await first.enqueue(make_card("+5541900000001"))
    await first.enqueue(make_card("+5541900000002"))
    await first.close()  # servidor cai na madrugada

    second = await CardStore.open(path)
    try:
        assert [c.status for c in await second.list()] == [CardStatus.PENDING] * 2
        notifier = RecordingNotifier()
        await dispatcher(second, notifier, Clock(sp(2026, 10, 27, 8, 30))).run_once()
        assert len(notifier.sent) == 2
    finally:
        await second.close()


async def test_card_preso_em_sending_volta_a_fila(tmp_path):
    path = tmp_path / "scheduler.sqlite"
    first = await CardStore.open(path)
    await first.enqueue(make_card())
    claimed = await first.claim_due(sp(2026, 10, 27, 8, 30))
    assert [c.status for c in claimed] == [CardStatus.SENDING]
    await first.close()  # processo morre no meio do envio

    second = await CardStore.open(path)
    try:
        assert await second.claim_due(sp(2026, 10, 27, 8, 31)) == []  # preso em SENDING
        assert await second.recover() == 1
        notifier = RecordingNotifier()
        await dispatcher(second, notifier, Clock(sp(2026, 10, 27, 8, 31))).run_once()
        assert len(notifier.sent) == 1
        assert (await second.list())[0].status is CardStatus.SENT
    finally:
        await second.close()


async def test_claim_e_atomico_entre_workers(tmp_path):
    path = tmp_path / "scheduler.sqlite"
    a, b = await CardStore.open(path), await CardStore.open(path)
    try:
        await a.enqueue(make_card())
        now = sp(2026, 10, 27, 8, 30)
        assert len(await a.claim_due(now)) == 1
        assert await b.claim_due(now) == []
    finally:
        await a.close()
        await b.close()


async def test_claim_perde_corrida_para_outro_worker(store, monkeypatch):
    await store.enqueue(make_card())
    original_execute = store._conn.execute

    async def execute(sql, params=()):
        if sql.startswith("UPDATE seller_cards SET status = 'SENDING'"):
            # outro worker reivindicou entre o SELECT e o UPDATE
            await original_execute("UPDATE seller_cards SET status = 'SENDING'")
        return await original_execute(sql, params)

    monkeypatch.setattr(store._conn, "execute", execute)
    assert await store.claim_due(sp(2026, 10, 27, 8, 30)) == []


async def test_get_inexistente(store):
    with pytest.raises(KeyError):
        await store.get(999)


async def test_open_em_memoria():
    s = await CardStore.open(":memory:")
    try:
        await s.enqueue(make_card())
        assert len(await s.list()) == 1
    finally:
        await s.close()


# ---------------------------------------------------------------- loop do worker


async def test_run_forever_recupera_entrega_e_sobrevive_a_tick_quebrado(store, monkeypatch, caplog):
    await store.enqueue(make_card())
    await store.claim_due(sp(2026, 10, 27, 8, 30))  # simula card preso em SENDING
    notifier = RecordingNotifier()
    d = CardDispatcher(
        store, notifier, clock=Clock(sp(2026, 10, 27, 8, 30)), retry_wait=wait_none(),
        interval_s=0,
    )
    original = d.run_once
    calls = {"n": 0}

    async def flaky_run_once() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("sqlite ocupado")
        return await original()

    monkeypatch.setattr(d, "run_once", flaky_run_once)
    with caplog.at_level(logging.WARNING):
        task = asyncio.create_task(d.run_forever())
        for _ in range(200):
            if notifier.sent:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(notifier.sent) == 1
    assert "devolvidos à fila" in caplog.text
    assert "tick do scheduler falhou" in caplog.text


# ---------------------------------------------------------------- webhook notifier


def _queued() -> QueuedCard:
    return QueuedCard(
        id=1, lead_id="+5541900000001", deliver_at=datetime(2026, 10, 27, 11, 30, tzinfo=UTC),
        prioridade="alta", bloqueado=False, texto="📋 CARD", status=CardStatus.SENDING,
        attempts=0, last_error=None, sent_at=None,
    )


async def test_webhook_notifier_envia_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200)

    n = WebhookNotifier(
        "https://hooks.exemplo/vendedor", recipient="+5541977776666",
        transport=httpx.MockTransport(handler),
    )
    await n.send(_queued())
    assert captured["url"] == "https://hooks.exemplo/vendedor"
    assert captured["body"] == {
        "lead_id": "+5541900000001",
        "prioridade": "alta",
        "bloqueado": False,
        "texto": "📋 CARD",
        "deliver_at": "2026-10-27T11:30:00+00:00",
        "to": "+5541977776666",
    }


async def test_webhook_notifier_http_erro_vira_notifier_error():
    n = WebhookNotifier("https://x", transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(NotifierError, match="HTTP 503"):
        await n.send(_queued())


async def test_webhook_notifier_falha_de_rede_vira_notifier_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("recusada", request=request)

    n = WebhookNotifier("https://x", transport=httpx.MockTransport(boom))
    with pytest.raises(NotifierError, match="ConnectError"):
        await n.send(_queued())


def test_build_notifier(monkeypatch):
    monkeypatch.setattr(settings, "seller_webhook_url", None)
    assert isinstance(main.build_notifier(), RecordingNotifier)
    monkeypatch.setattr(settings, "seller_webhook_url", SecretStr("https://hooks.exemplo/v"))
    assert isinstance(main.build_notifier(), WebhookNotifier)


# ---------------------------------------------------------------- integração grafo/API


@pytest.fixture(scope="module")
def predictor():
    p = LDMPredictor(backend="sklearn")
    p.fit(generate_dataset(2_000, seed=5))
    return p


async def test_no_do_grafo_enfileira_card(store, predictor):
    graph = build_graph(
        InMemorySaver(serde=checkpoint_serde()), TemplateLLMClient(), predictor, store
    )
    phone = "+5541955554444"
    result = await graph.ainvoke(
        initial_state("Quero um SUV até 140 mil", phone, VehicleAppraisalInput(**TRADE_IN)),
        {"configurable": {"thread_id": phone}},
    )
    (queued,) = await store.list(CardStatus.PENDING)
    assert queued.lead_id == phone
    assert queued.texto == result["seller_briefing_card"].texto
    assert queued.deliver_at.astimezone(SP).time().isoformat() == "08:30:00"


async def test_falha_ao_enfileirar_nao_derruba_resposta(store, predictor, monkeypatch, caplog):
    async def broken(card, **kw):
        raise RuntimeError("disco cheio")

    monkeypatch.setattr(store, "enqueue", broken)
    graph = build_graph(
        InMemorySaver(serde=checkpoint_serde()), TemplateLLMClient(), predictor, store
    )
    phone = "+5541955553333"
    with caplog.at_level(logging.ERROR):
        result = await graph.ainvoke(
            initial_state("Quero um SUV até 140 mil", phone, VehicleAppraisalInput(**TRADE_IN)),
            {"configurable": {"thread_id": phone}},
        )
    assert result["final_reply"]
    assert result["seller_briefing_card"] is not None  # card preservado no checkpoint
    assert "falha ao enfileirar" in caplog.text


def test_lifespan_sobe_worker_e_enfileira_via_webhook(monkeypatch, predictor, tmp_path):
    monkeypatch.setattr(settings, "checkpoint_backend", "memory")
    monkeypatch.setattr(settings, "scheduler_db_path", str(tmp_path / "s.sqlite"))
    monkeypatch.setattr(settings, "seller_webhook_url", None)
    monkeypatch.setattr(main, "get_llm_client", TemplateLLMClient)
    monkeypatch.setattr(main, "get_langfuse_handler", lambda: None)
    monkeypatch.setattr(main, "build_ldm_predictor", lambda: predictor)

    with TestClient(main.app) as client:
        resp = client.post(
            "/webhook",
            json={"phone": "+5541900009999", "text": "Quero um SUV até 140 mil",
                  "trade_in": TRADE_IN},
        )
        assert resp.status_code == 200
        queue = main.app.state.card_queue
        cards = client.portal.call(queue.list)
        assert [c.lead_id for c in cards] == ["+5541900009999"]
        assert cards[0].status is CardStatus.PENDING  # só sai às 08h30
    # depois do shutdown, a fila continua em disco
    assert (tmp_path / "s.sqlite").exists()


def test_lifespan_sem_ldm_nao_sobe_scheduler(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_backend", "memory")
    monkeypatch.setattr(settings, "ldm_enabled", False)
    monkeypatch.setattr(main, "get_llm_client", TemplateLLMClient)
    monkeypatch.setattr(main, "get_langfuse_handler", lambda: None)
    with TestClient(main.app):
        assert main.app.state.card_queue is None


def test_delta_fuso_constante():
    # Brasil sem horário de verão desde 2019: 08h30 SP é sempre 11h30 UTC.
    for month in (1, 4, 7, 10, 12):
        assert proximo_envio(sp(2026, month, 1, 0, 0)).utcoffset() == timedelta(hours=-3)

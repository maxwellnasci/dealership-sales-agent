"""Worker assíncrono que entrega os cards às 08h30 (America/Sao_Paulo).

A cada `interval_s` (padrão 60 s) o worker reivindica os cards com
`deliver_at <= agora` e os envia pelo `Notifier`, com retry exponencial de
até 3 tentativas por card. Esgotadas as tentativas, o card fica `FAILED` com o
erro registrado — visível para reprocessamento, nunca descartado em silêncio.

O horário de entrega já vem no card (`SellerCard.envio_agendado_para`, as
próximas 08h30 de Curitiba a partir da geração), então o worker só compara
instantes em UTC: não há aritmética de fuso aqui.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from dealership_agent.scheduler.notifier import Notifier, NotifierError
from dealership_agent.scheduler.store import CardStore, QueuedCard

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CardDispatcher:
    def __init__(
        self,
        store: CardStore,
        notifier: Notifier,
        *,
        clock: Callable[[], datetime] = _utcnow,
        retry_wait: wait_base | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        interval_s: float = 60.0,
    ) -> None:
        self.store = store
        self.notifier = notifier
        self.clock = clock
        # 2 s, 4 s entre as tentativas (teto de 30 s): cabe folgado no tick de 1 min.
        self.retry_wait = retry_wait or wait_exponential(multiplier=2, min=2, max=30)
        self.sleep = sleep
        self.interval_s = interval_s

    async def _deliver(self, card: QueuedCard) -> None:
        attempts = 0
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(MAX_ATTEMPTS),
                wait=self.retry_wait,
                sleep=self.sleep,
                retry=retry_if_exception_type(NotifierError),
            ):
                with attempt:
                    attempts += 1
                    await self.notifier.send(card)
        except RetryError as exc:
            error = str(exc.last_attempt.exception())
            logger.error("card %s FAILED após %d tentativas: %s", card.id, attempts, error)
            await self.store.mark_failed(card.id, attempts=attempts, error=error, now=self.clock())
            return
        except Exception as exc:  # erro não-transitório do notifier: falha sem retry
            logger.exception("card %s FAILED (erro inesperado)", card.id)
            await self.store.mark_failed(
                card.id, attempts=attempts, error=f"{type(exc).__name__}: {exc}", now=self.clock()
            )
            return
        await self.store.mark_sent(card.id, attempts=attempts, sent_at=self.clock())

    async def run_once(self) -> int:
        """Um tick: entrega tudo que está vencido. Devolve quantos cards processou."""
        due = await self.store.claim_due(self.clock())
        for card in due:
            await self._deliver(card)
        return len(due)

    async def run_forever(self) -> None:
        """Loop do worker. Um tick que falha é logado e o loop segue."""
        recovered = await self.store.recover()
        if recovered:
            logger.warning("%d card(s) em SENDING devolvidos à fila após restart", recovered)
        while True:
            try:
                await self.run_once()
            except Exception:
                logger.exception("tick do scheduler falhou; tentando de novo no próximo")
            await asyncio.sleep(self.interval_s)

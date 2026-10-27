"""Despacho do card para o vendedor (webhook / WhatsApp do vendedor).

`Notifier` é a porta; dois adaptadores:

  - `WebhookNotifier`: POST JSON num endpoint (ex: gateway de WhatsApp Business,
    n8n, Slack incoming webhook). Status != 2xx é falha (vai para o retry).
  - `RecordingNotifier`: mock que só registra em memória/log — padrão quando
    nenhum webhook está configurado, e usado nos testes.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Protocol

import httpx

from dealership_agent.scheduler.store import QueuedCard

logger = logging.getLogger(__name__)


class NotifierError(Exception):
    """Falha de entrega (transitória ou não): o dispatcher aplica o retry."""


class Notifier(Protocol):
    async def send(self, card: QueuedCard) -> None: ...


def card_payload(card: QueuedCard) -> dict:
    return {
        "lead_id": card.lead_id,
        "prioridade": card.prioridade,
        "bloqueado": card.bloqueado,
        "texto": card.texto,
        "deliver_at": card.deliver_at.isoformat(),
    }


class WebhookNotifier:
    def __init__(
        self,
        url: str,
        *,
        recipient: str | None = None,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = url
        self._recipient = recipient
        self._timeout = timeout_s
        self._transport = transport

    async def send(self, card: QueuedCard) -> None:
        payload = card_payload(card) | {"to": self._recipient}
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                resp = await client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            raise NotifierError(f"{type(exc).__name__}: {exc}") from exc
        if not resp.is_success:
            raise NotifierError(f"webhook respondeu HTTP {resp.status_code}")


class RecordingNotifier:
    """Mock de despacho: guarda (card, momento) e loga. Sem rede."""

    def __init__(self) -> None:
        self.sent: list[tuple[QueuedCard, datetime]] = []

    async def send(self, card: QueuedCard) -> None:
        self.sent.append((card, datetime.now(UTC)))
        logger.info("card %s (lead %s) entregue via mock", card.id, card.lead_id)

"""Fila durável dos Cards Executivos (SQLite via aiosqlite).

Cada card gerado durante o dia/madrugada vira uma linha `PENDING` com o
horário de entrega (`deliver_at`, em UTC). O worker entrega e marca `SENT`
(com `sent_at`) ou, esgotado o retry, `FAILED` (com `last_error`).

Garantias:
  - **Nada se perde num restart:** a fila está em disco; a linha só sai de
    `PENDING` quando o worker a reivindica.
  - **Reivindicação atômica:** antes de enviar, a linha passa a `SENDING` num
    `UPDATE ... WHERE status='PENDING'`, então dois workers nunca enviam o
    mesmo card. Se o processo morrer no meio do envio, `recover()` (chamado no
    startup) devolve `SENDING` para `PENDING`: entrega *at-least-once* —
    preferimos um card duplicado a um card perdido.
  - **Idempotência por (lead, horário):** o mesmo lead recalculado várias vezes
    antes das 08h30 mantém UMA linha, com o texto mais recente.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import aiosqlite

from mxos_ldm.sales_bridge import SellerCard

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seller_cards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id     TEXT NOT NULL,
    deliver_at  TEXT NOT NULL,          -- ISO-8601 em UTC (ordenável como texto)
    prioridade  TEXT NOT NULL,
    bloqueado   INTEGER NOT NULL,
    texto       TEXT NOT NULL,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    sent_at     TEXT,
    UNIQUE (lead_id, deliver_at)
);
CREATE INDEX IF NOT EXISTS ix_seller_cards_due ON seller_cards (status, deliver_at);
"""


class CardStatus(StrEnum):
    PENDING = "PENDING"
    SENDING = "SENDING"  # transitório: reivindicado por um worker
    SENT = "SENT"
    FAILED = "FAILED"


@dataclass(frozen=True)
class QueuedCard:
    id: int
    lead_id: str
    deliver_at: datetime
    prioridade: str
    bloqueado: bool
    texto: str
    status: CardStatus
    attempts: int
    last_error: str | None
    sent_at: datetime | None


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("datetime sem fuso: a fila só aceita horários com tzinfo")
    return dt.astimezone(UTC).isoformat()


def _row_to_card(row: aiosqlite.Row) -> QueuedCard:
    return QueuedCard(
        id=row["id"],
        lead_id=row["lead_id"],
        deliver_at=datetime.fromisoformat(row["deliver_at"]),
        prioridade=row["prioridade"],
        bloqueado=bool(row["bloqueado"]),
        texto=row["texto"],
        status=CardStatus(row["status"]),
        attempts=row["attempts"],
        last_error=row["last_error"],
        sent_at=datetime.fromisoformat(row["sent_at"]) if row["sent_at"] else None,
    )


class CardStore:
    """Fila de cards sobre uma conexão aiosqlite. Use `CardStore.open(path)`."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = aiosqlite.Row

    @classmethod
    async def open(cls, path: str | Path) -> CardStore:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(path))
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.executescript(_SCHEMA)
        await conn.commit()
        return cls(conn)

    async def close(self) -> None:
        await self._conn.close()

    async def enqueue(self, card: SellerCard, *, now: datetime | None = None) -> None:
        """Enfileira (ou atualiza, se ainda PENDING) o card do lead para aquele horário.

        Card já enviado/falho para o mesmo (lead, horário) não é reaberto.
        """
        ts = _iso(now or datetime.now(UTC))
        await self._conn.execute(
            """
            INSERT INTO seller_cards
                (lead_id, deliver_at, prioridade, bloqueado, texto, status,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?)
            ON CONFLICT (lead_id, deliver_at) DO UPDATE SET
                prioridade = excluded.prioridade,
                bloqueado  = excluded.bloqueado,
                texto      = excluded.texto,
                updated_at = excluded.updated_at
            WHERE seller_cards.status = 'PENDING'
            """,
            (
                card.lead_id,
                _iso(card.envio_agendado_para),
                card.prioridade,
                int(card.bloqueado),
                card.texto,
                ts,
                ts,
            ),
        )
        await self._conn.commit()

    async def recover(self) -> int:
        """Devolve à fila os cards que ficaram em SENDING (processo morreu no envio)."""
        cur = await self._conn.execute(
            "UPDATE seller_cards SET status = 'PENDING' WHERE status = 'SENDING'"
        )
        await self._conn.commit()
        return cur.rowcount

    async def claim_due(self, now: datetime) -> list[QueuedCard]:
        """Reivindica (PENDING -> SENDING) todos os cards com `deliver_at <= now`.

        Cards atrasados (servidor fora do ar às 08h30) também saem: atrasado é
        melhor que perdido.
        """
        cur = await self._conn.execute(
            "SELECT id FROM seller_cards WHERE status = 'PENDING' AND deliver_at <= ?"
            " ORDER BY deliver_at, id",
            (_iso(now),),
        )
        ids = [row["id"] for row in await cur.fetchall()]
        claimed: list[QueuedCard] = []
        for card_id in ids:
            upd = await self._conn.execute(
                "UPDATE seller_cards SET status = 'SENDING', updated_at = ?"
                " WHERE id = ? AND status = 'PENDING'",
                (_iso(now), card_id),
            )
            if upd.rowcount == 1:
                claimed.append(await self.get(card_id))
        await self._conn.commit()
        return claimed

    async def mark_sent(self, card_id: int, *, attempts: int, sent_at: datetime) -> None:
        await self._conn.execute(
            "UPDATE seller_cards SET status = 'SENT', attempts = attempts + ?, sent_at = ?,"
            " last_error = NULL, updated_at = ? WHERE id = ?",
            (attempts, _iso(sent_at), _iso(sent_at), card_id),
        )
        await self._conn.commit()

    async def mark_failed(
        self, card_id: int, *, attempts: int, error: str, now: datetime
    ) -> None:
        await self._conn.execute(
            "UPDATE seller_cards SET status = 'FAILED', attempts = attempts + ?,"
            " last_error = ?, updated_at = ? WHERE id = ?",
            (attempts, error[:500], _iso(now), card_id),
        )
        await self._conn.commit()

    async def get(self, card_id: int) -> QueuedCard:
        cur = await self._conn.execute("SELECT * FROM seller_cards WHERE id = ?", (card_id,))
        row = await cur.fetchone()
        if row is None:
            raise KeyError(card_id)
        return _row_to_card(row)

    async def list(self, status: CardStatus | None = None) -> list[QueuedCard]:
        if status is None:
            cur = await self._conn.execute("SELECT * FROM seller_cards ORDER BY id")
        else:
            cur = await self._conn.execute(
                "SELECT * FROM seller_cards WHERE status = ? ORDER BY id", (status.value,)
            )
        return [_row_to_card(r) for r in await cur.fetchall()]

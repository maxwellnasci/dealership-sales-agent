"""Tool de agenda de test-drive (mock determinístico, em memória).

Gera horários disponíveis a partir de uma data-base e "confirma" um slot. Numa
implementação real isto bateria no Google Calendar / agenda do CRM; aqui é
puro para a demo rodar sem dependências. O retry fica de fora de propósito:
reservar um horário é escrita com efeito colateral e não deve ser re-tentada
cegamente (mesma disciplina do estoque, invertida).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from dealership_agent.schemas.tools import TestDriveSlot, Vehicle

_BUSINESS_HOURS = [time(9, 0), time(11, 0), time(14, 0), time(16, 0)]


def available_slots(
    vehicle: Vehicle, *, from_date: datetime | None = None, days: int = 2
) -> list[TestDriveSlot]:
    """Lista horários livres para test-drive nos próximos `days` dias úteis."""
    base = (from_date or datetime(2026, 10, 19, 9, 0)).date()
    slots: list[TestDriveSlot] = []
    for day_offset in range(days):
        day = base + timedelta(days=day_offset)
        for hour in _BUSINESS_HOURS:
            starts = datetime.combine(day, hour)
            slots.append(
                TestDriveSlot(
                    slot_id=f"{vehicle.vehicle_id}-{starts:%Y%m%d%H%M}",
                    vehicle_id=vehicle.vehicle_id,
                    vehicle_label=vehicle.short_label(),
                    starts_at=starts,
                    confirmed=False,
                )
            )
    return slots


def book_slot(slot: TestDriveSlot) -> TestDriveSlot:
    """Confirma um horário. Idempotente: confirmar de novo devolve o mesmo slot."""
    return slot.model_copy(update={"confirmed": True})

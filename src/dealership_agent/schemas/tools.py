"""Schemas tipados das tools de negócio.

Preços, parcelas, veículos e horários SÓ existem como instâncias destes
modelos, produzidos pelas tools (tools/*.py). O LLM nunca gera esses números:
ele apenas verbaliza o que já está tipado aqui. Essa é a outra metade da
estratégia anti-alucinação (a primeira é o `LeadProfile`).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Vehicle(BaseModel):
    """Um veículo do estoque."""

    vehicle_id: str
    make: str
    model: str
    year: int
    price: float = Field(description="Preço à vista em reais.")
    mileage_km: int = Field(description="Quilometragem.")
    body_type: str
    color: str

    def short_label(self) -> str:
        return f"{self.make} {self.model} {self.year}"


class InventoryQuery(BaseModel):
    """Parâmetros de busca no estoque. Extraídos do perfil do lead."""

    make: str | None = None
    model: str | None = None
    body_type: str | None = None
    max_price: float | None = None


class FinancingQuote(BaseModel):
    """Simulação de financiamento — números 100% determinísticos (tools)."""

    vehicle_id: str
    vehicle_label: str
    price: float
    down_payment: float = Field(description="Entrada em reais.")
    term_months: int = Field(description="Prazo em meses.")
    monthly_rate: float = Field(description="Taxa de juros mensal (fração, ex: 0.019).")
    monthly_payment: float = Field(description="Parcela mensal em reais.")
    total_paid: float = Field(description="Total pago ao fim do contrato.")


class TestDriveSlot(BaseModel):
    """Horário disponível / confirmado para test-drive."""

    slot_id: str
    vehicle_id: str
    vehicle_label: str
    starts_at: datetime
    confirmed: bool = False

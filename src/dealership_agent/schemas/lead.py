"""Perfil do lead e estágios de qualificação.

`LeadProfile` é o estado de negócio que o agente acumula ao longo da conversa.
Cada campo é opcional porque a informação chega aos poucos (WhatsApp, uma
mensagem por vez); o reducer `merge_lead_profile` (graph/state.py) combina o
que já se sabia com o que a mensagem nova revelou, sem nunca apagar um dado já
coletado.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class QualificationStage(StrEnum):
    """Onde o lead está no funil. Dirige o tom e a próxima pergunta do agente."""

    GREETING = "greeting"          # primeiro contato, ainda sem dados
    DISCOVERY = "discovery"        # coletando necessidade (modelo, orçamento)
    PRESENTING = "presenting"      # apresentando veículos do estoque
    SCHEDULING = "scheduling"      # agendando test-drive
    FINANCING = "financing"        # simulando financiamento
    HANDOFF = "handoff"            # transbordo para vendedor humano
    CLOSED = "closed"              # conversa encerrada


class LeadProfile(BaseModel):
    """Tudo que sabemos sobre o lead. Preenchido incrementalmente.

    É a fonte da verdade do negócio: o agente só pode afirmar sobre o lead o
    que estiver aqui (ou sobre veículos o que vier das tools). Isso é metade
    da estratégia anti-alucinação.
    """

    name: str | None = Field(default=None, description="Primeiro nome do lead, se informado.")
    desired_make: str | None = Field(default=None, description="Marca de interesse (ex: Toyota).")
    desired_model: str | None = Field(
        default=None, description="Modelo de interesse (ex: Corolla)."
    )
    budget_max: float | None = Field(
        default=None,
        description="Orçamento máximo em reais, se o lead mencionou um teto.",
    )
    body_type: str | None = Field(
        default=None, description="Carroceria desejada: sedan, suv, hatch, picape..."
    )
    purpose: str | None = Field(
        default=None, description="Uso pretendido: família, trabalho, cidade, viagem..."
    )
    trade_in: bool | None = Field(
        default=None, description="Se o lead tem um carro na troca."
    )
    phone: str | None = Field(default=None, description="Telefone/identificador do lead.")

    def merged_with(self, other: LeadProfile) -> LeadProfile:
        """Retorna um novo perfil preferindo os campos preenchidos de `other`
        sobre os atuais, sem jamais sobrescrever um valor existente por None."""
        base = self.model_dump()
        for key, value in other.model_dump().items():
            if value is not None:
                base[key] = value
        return LeadProfile(**base)

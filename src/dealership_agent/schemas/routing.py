"""Decisão de roteamento do supervisor (structured output).

O nó supervisor (router) pede ao LLM uma saída **tipada** `RouteDecision`.
Como a saída é validada por Pydantic, o modelo não consegue "inventar" um
destino inexistente: ou devolve um `Intent` válido, ou a validação falha e o
grafo cai num fallback seguro. Esse contrato é o que torna o roteamento
confiável.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Intent(StrEnum):
    """Intenções que o supervisor sabe rotear. Cada valor mapeia 1:1 para um
    nó do grafo (ver graph/builder.py::INTENT_TO_NODE)."""

    QUALIFY = "qualify"              # coletar/atualizar necessidade do lead
    SEARCH_INVENTORY = "search_inventory"  # buscar veículos no estoque
    SCHEDULE_TEST_DRIVE = "schedule_test_drive"  # agendar test-drive
    FINANCING = "financing"          # simular financiamento
    HANDOFF = "handoff"              # falar com vendedor humano
    SMALL_TALK = "small_talk"        # saudação / assunto fora de escopo


class RouteDecision(BaseModel):
    """Saída estruturada do supervisor."""

    intent: Intent = Field(description="A próxima ação a tomar para esta mensagem do lead.")
    rationale: str = Field(
        default="",
        description="Uma frase curta explicando a escolha (observabilidade; não vai ao lead).",
    )

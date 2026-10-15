"""Contrato do estado que trafega entre os nós do grafo.

Dois campos têm reducer (semântica de acumulação entre nós e entre turnos):

  - `messages`: histórico da conversa, reducer `add_messages` (LangGraph).
  - `lead_profile`: perfil do lead, reducer `merge_lead_profile` — merge
    incremental que nunca apaga um dado já coletado.

O resto é "por turno": carregado do checkpoint, sobrescrito pelo turno atual.
O checkpointer persiste o estado inteiro por `thread_id` (= telefone do lead),
então a memória entre mensagens de WhatsApp é literalmente este TypedDict
ressuscitado a cada mensagem.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.tools import FinancingQuote, TestDriveSlot, Vehicle


def merge_lead_profile(old: LeadProfile | None, new: LeadProfile | None) -> LeadProfile:
    """Reducer do `lead_profile`.

    Combina o perfil acumulado (`old`) com o que o turno atual descobriu
    (`new`), preferindo campos preenchidos de `new` mas preservando tudo que
    já se sabia. Nunca "esquece" um dado por causa de uma mensagem que não o
    repetiu — crítico numa conversa de WhatsApp fragmentada.
    """
    if old is None:
        return new or LeadProfile()
    if new is None:
        return old
    return old.merged_with(new)


class GraphState(TypedDict, total=False):
    """Estado do grafo de vendas."""

    # --- Acumulados (reducers) -------------------------------------------
    messages: Annotated[list, add_messages]
    lead_profile: Annotated[LeadProfile, merge_lead_profile]

    # --- Por turno --------------------------------------------------------
    route: str | None               # Intent decidida pelo supervisor (observabilidade + aresta)
    retrieved_vehicles: list[Vehicle]   # resultado da última busca de estoque
    financing_quote: FinancingQuote | None
    test_drive_slot: TestDriveSlot | None
    final_reply: str | None         # resposta natural já validada pelos guardrails
    guardrail_flags: list[str]      # o que o validador barrou/ajustou (observabilidade)


def initial_state(message: str, phone: str) -> dict:
    """Estado de entrada para uma mensagem nova do lead.

    Só injeta a mensagem do usuário e o telefone. `messages` e `lead_profile`
    vêm do checkpoint (reducers acumulam); os demais campos são recomputados
    pelo turno, então nem precisam ser zerados explicitamente aqui.
    """
    return {
        "messages": [{"role": "user", "content": message}],
        "lead_profile": LeadProfile(phone=phone),
    }

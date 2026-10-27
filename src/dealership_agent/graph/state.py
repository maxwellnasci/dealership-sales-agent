"""Contrato do estado que trafega entre os nós do grafo.

Dois campos têm reducer (semântica de acumulação entre nós e entre turnos):

  - `messages`: histórico da conversa, reducer `add_messages` (LangGraph).
  - `lead_profile`: perfil do lead, reducer `merge_lead_profile` — merge
    incremental que nunca apaga um dado já coletado.

O resto é "por turno": carregado do checkpoint, sobrescrito pelo turno atual —
exceto os campos do LDM (`ldm_*` e `seller_briefing_card`), que persistem
entre turnos: o carro da troca não muda a cada "oi" e o card das 08h30 é lido
do checkpoint. Eles nunca chegam à resposta do lead (são só para o vendedor).
O checkpointer persiste o estado inteiro por `thread_id` (= telefone do lead),
então a memória entre mensagens de WhatsApp é literalmente este TypedDict
ressuscitado a cada mensagem.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.tools import FinancingQuote, TestDriveSlot, Vehicle
from mxos_ldm.sales_bridge import SellerCard
from mxos_ldm.schemas import PredictionOutput, VehicleAppraisalInput


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

    # --- LDM (persistem entre turnos; só para o vendedor humano) ----------
    ldm_vehicle_input: VehicleAppraisalInput | None  # carro da troca (FIPE vem de fonte externa)
    ldm_prediction: PredictionOutput | None
    seller_briefing_card: SellerCard | None


def initial_state(
    message: str, phone: str, ldm_vehicle_input: VehicleAppraisalInput | None = None
) -> dict:
    """Estado de entrada para uma mensagem nova do lead.

    `messages` e `lead_profile` ACUMULAM via reducer (memória real entre
    mensagens de WhatsApp). Todos os demais campos são "por turno" e são
    zerados EXPLICITAMENTE aqui: como o checkpointer ressuscita o estado
    inteiro por `thread_id`, um `financing_quote`/`test_drive_slot`/
    `guardrail_flags` de um turno anterior vazaria para um turno seguinte que
    não os recompute (ex: o lead volta dias depois só pra dizer "oi" e veria
    uma simulação/agendamento velho). Resetar na entrada elimina esse vazamento.

    `ldm_vehicle_input` só entra no dict quando informado: omitir a chave
    preserva o carro da troca já persistido no checkpoint.
    """
    state = {
        "messages": [{"role": "user", "content": message}],
        "lead_profile": LeadProfile(phone=phone),
        "route": None,
        "retrieved_vehicles": [],
        "financing_quote": None,
        "test_drive_slot": None,
        "final_reply": None,
        "guardrail_flags": [],
    }
    if ldm_vehicle_input is not None:
        state["ldm_vehicle_input"] = ldm_vehicle_input
    return state

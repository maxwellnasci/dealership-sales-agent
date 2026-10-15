"""Nó de qualificação: extrai/atualiza o perfil do lead.

Usa structured output para transformar a mensagem livre do cliente em campos
de `LeadProfile`. O retorno vai pro reducer `merge_lead_profile`, que acumula
sem apagar nada já coletado.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from dealership_agent.graph.state import GraphState
from dealership_agent.llm import LLMClient
from dealership_agent.schemas.lead import LeadProfile


def _last_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        role = getattr(msg, "type", None)
        if content and role in (None, "human"):
            return str(content)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


async def qualify_lead_node(state: GraphState, *, llm: LLMClient) -> dict:
    text = _last_user_text(state.get("messages", []))
    current = state.get("lead_profile") or LeadProfile()
    extracted = await llm.extract_lead(text, current)
    # O reducer faz o merge; devolvemos só o delta extraído.
    return {"lead_profile": extracted}

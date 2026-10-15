"""Nó de transbordo humano (human-in-the-loop com `interrupt`).

Quando o lead pede um vendedor, quer negociar preço ou o agente perde a
confiança, o grafo **pausa de verdade** via `interrupt()`: a execução para, o
checkpoint guarda o ponto exato, e o `interrupt` devolve ao chamador um
payload com o contexto do lead. O vendedor humano responde e o chamador retoma
com `Command(resume=<texto do vendedor>)`; o valor do resume vira o retorno de
`interrupt()`, e a mensagem do vendedor segue ao lead como resposta do turno.

É o padrão nativo de HITL do LangGraph — diferente de um "confirme no próximo
turno": aqui o MESMO turno fica suspenso, retomável a qualquer momento graças
ao checkpointer, sem perder estado.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from dealership_agent.config import settings
from dealership_agent.graph.state import GraphState
from dealership_agent.llm import LLMClient
from dealership_agent.schemas.lead import LeadProfile


async def handoff_human_node(state: GraphState, *, llm: LLMClient) -> dict:
    lead = state.get("lead_profile") or LeadProfile()
    messages = state.get("messages", [])
    last_user = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if content:
            last_user = str(content)
            break

    # PAUSA: devolve contexto ao operador e espera o resume.
    human_reply = interrupt(
        {
            "reason": "handoff_requested",
            "assigned_to": settings.sales_rep_name,
            "lead": lead.model_dump(),
            "last_message": last_user,
            "prompt_for_human": (
                f"{settings.sales_rep_name}, um lead precisa de você. "
                "Responda e sua mensagem volta direto para o cliente."
            ),
        }
    )

    reply_text = str(human_reply).strip() or (
        "Um de nossos vendedores vai te chamar agora mesmo por aqui. 🙂"
    )
    return {
        "final_reply": reply_text,
        "messages": [AIMessage(content=reply_text)],
        "guardrail_flags": [],
    }

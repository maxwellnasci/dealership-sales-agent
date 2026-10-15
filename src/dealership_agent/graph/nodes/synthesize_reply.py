"""Nó de síntese da resposta.

Estratégia em duas etapas para resposta natural E confiável:

1. Monta um RASCUNHO determinístico a partir só dos fatos do estado (perfil do
   lead, veículos recuperados, simulação, agendamento). Por construção, o
   rascunho nunca inventa número.
2. Pede ao LLM para reescrever o rascunho de forma natural e calorosa, sem
   adicionar fatos. Em seguida o guardrail (`validate_reply`) confere que o
   texto final não introduziu nenhum valor monetário fora dos fatos; se
   introduziu, cai de volta no rascunho seguro.

Assim o pior caso é uma resposta correta porém mais "seca" — nunca uma mentira
de preço para o cliente.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from dealership_agent.config import settings
from dealership_agent.graph.state import GraphState
from dealership_agent.guardrails.validator import validate_reply
from dealership_agent.llm import LLMClient
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.routing import Intent


def _fmt_brl(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def build_draft(state: GraphState) -> str:
    """Rascunho factual e seguro (nunca alucina)."""
    lead: LeadProfile = state.get("lead_profile") or LeadProfile()
    route = state.get("route")
    vehicles = state.get("retrieved_vehicles") or []
    quote = state.get("financing_quote")
    slot = state.get("test_drive_slot")
    greeting = f"Oi{(' ' + lead.name) if lead.name else ''}!"

    if route == Intent.SMALL_TALK.value:
        return (
            f"{greeting} Sou o assistente da revenda. Posso te ajudar a encontrar "
            "um carro, simular um financiamento ou agendar um test-drive. O que você procura?"
        )

    if route == Intent.FINANCING.value and quote is not None:
        return (
            f"{greeting} Para o {quote.vehicle_label} (à vista {_fmt_brl(quote.price)}), "
            f"com entrada de {_fmt_brl(quote.down_payment)} em {quote.term_months}x, "
            f"a parcela fica em {_fmt_brl(quote.monthly_payment)} "
            f"(total de {_fmt_brl(quote.total_paid)}). Quer que eu já agende um test-drive?"
        )

    if route == Intent.SCHEDULE_TEST_DRIVE.value and slot is not None:
        return (
            f"{greeting} Consigo agendar seu test-drive do {slot.vehicle_label} para "
            f"{slot.starts_at:%d/%m às %Hh%M}. Esse único horário fica bom pra você? "
            "Se não der, me avisa que eu verifico outra data na agenda."
        )

    if vehicles:
        def _line(v) -> str:
            km = f"{v.mileage_km:,}".replace(",", ".")  # 61,000 -> 61.000 (pt-BR)
            return f"- {v.short_label()} ({v.body_type}, {km} km) por {_fmt_brl(v.price)}"

        listing = "\n".join(_line(v) for v in vehicles[: settings.max_vehicle_suggestions])
        return (
            f"{greeting} Separei estas opções que combinam com o que você procura:\n{listing}\n"
            "Quer simular o financiamento de algum ou agendar um test-drive?"
        )

    # Descoberta: ainda falta informação.
    missing = []
    if not lead.desired_make and not lead.desired_model and not lead.body_type:
        missing.append("que tipo de carro você procura (modelo ou carroceria)")
    if lead.budget_max is None:
        missing.append("qual faixa de preço cabe no seu orçamento")
    ask = " e ".join(missing) if missing else "um pouco mais sobre o que você precisa"
    return f"{greeting} Para eu te ajudar melhor, me conta {ask}?"


_PERSONA = (
    "Você é {agent_name}, vendedor consultivo de uma revenda de veículos, atendendo por "
    "WhatsApp. Se for se apresentar, use o nome {agent_name} (nunca um placeholder como "
    "'[Seu Nome]'). Tom caloroso, objetivo, em português brasileiro. Reescreva o RASCUNHO "
    "abaixo de forma natural e humana. REGRA ABSOLUTA: não adicione, altere ou invente NENHUM "
    "número, preço, parcela, data ou modelo que não esteja no rascunho. Mantenha todos os "
    "valores idênticos. REGRA DE AGENDA: só mencione dia, data ou horário se houver uma data/hora "
    "de test-drive JÁ no rascunho — e, nesse caso, ofereça EXATAMENTE aquela, sem alternativas. "
    "Se o rascunho NÃO traz data/hora, NUNCA proponha um dia ou horário por conta própria (nada "
    "de 'que tal sexta às 15h', 'amanhã', 'hoje'): apenas convide o lead a agendar, que o horário "
    "concreto é definido só quando o agendamento acontece.\n\n"
    "RASCUNHO:\n{draft}"
)


async def synthesize_reply_node(state: GraphState, *, llm: LLMClient) -> dict:
    vehicles = state.get("retrieved_vehicles") or []
    quote = state.get("financing_quote")
    slot = state.get("test_drive_slot")
    draft = build_draft(state)

    try:
        persona = _PERSONA.format(agent_name=settings.agent_name, draft=draft)
        natural = await llm.compose_reply(persona, state.get("messages", []))
        natural = (natural or "").strip() or draft
    except Exception:
        natural = draft

    flags = validate_reply(natural, vehicles, quote, slot)
    final = draft if flags else natural  # alucinou valor -> volta ao rascunho seguro

    return {
        "final_reply": final,
        "messages": [AIMessage(content=final)],
        "guardrail_flags": flags,
    }

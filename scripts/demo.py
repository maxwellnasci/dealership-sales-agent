"""Demo de terminal: roda conversas ponta a ponta pelo grafo.

Usa o LLM real se houver OPENAI_API_KEY; senão cai no fallback offline
determinístico (TemplateLLMClient) — a demo sempre roda. Demonstra:
  - memória entre mensagens (mesma thread = mesmo lead);
  - busca de estoque + financiamento + agendamento (tools tipadas);
  - human-in-the-loop: pausa no handoff e retomada com a fala do vendedor.

Rodar:  python scripts/demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dealership_agent.config import settings  # noqa: E402
from dealership_agent.graph.builder import build_graph  # noqa: E402
from dealership_agent.graph.state import initial_state  # noqa: E402
from dealership_agent.llm import get_llm_client  # noqa: E402
from dealership_agent.persistence.checkpointer import checkpoint_serde  # noqa: E402

SEED = json.loads((ROOT / "data" / "seed_conversations.json").read_text(encoding="utf-8"))


def _banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


async def run_conversation(graph, convo: dict) -> None:
    _banner(f"CONVERSA: {convo['title']}  (lead {convo['phone']})")
    config = {"configurable": {"thread_id": convo["phone"]}}

    for text in convo["turns"]:
        print(f"\n🧑 Lead: {text}")
        result = await graph.ainvoke(initial_state(text, convo["phone"]), config)

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"\n⏸️  [HANDOFF] Grafo pausado. Operador: {payload['assigned_to']}")
            print(f"    Contexto do lead: {payload['lead']}")
            rep_text = (
                "Oi! Aqui é o Carlos, da revenda. "
                "Consigo R$ 3.000 de desconto à vista — fechamos?"
            )
            print(f"\n🧑‍💼 Vendedor (humano): {rep_text}")
            result = await graph.ainvoke(Command(resume=rep_text), config)

        reply = result.get("final_reply")
        print(f"\n🤖 Agente: {reply}")
        if result.get("guardrail_flags"):
            print(f"   ⚠️  guardrail: {result['guardrail_flags']}")


async def main() -> None:
    mode = "LLM real" if settings.llm_configured else "fallback offline (sem API key)"
    print(f"Modo do LLM: {mode}")
    graph = build_graph(InMemorySaver(serde=checkpoint_serde()), get_llm_client())
    for convo in SEED:
        await run_conversation(graph, convo)
    print("\n✅ Demo concluída.\n")


if __name__ == "__main__":
    asyncio.run(main())

"""Benchmark: roteador clássico (LLM generativo) vs roteador Jev.

Roda 3 mensagens de teste contra os dois roteadores e imprime, para cada
um: intenção escolhida, latência (ms) e custo estimado em tokens.

- Clássico: `LLMClient.decide_route` (structured output generativo). Sem
  `OPENAI_API_KEY`, cai no `TemplateLLMClient` offline — a latência medida
  é local e os tokens são **estimados** pelo tamanho do prompt.
- Jev: `JevRouter` (typesafe/jev-1.14 via OpenRouter). Sem
  `OPENROUTER_API_KEY`, delega ao fallback clássico (indicado na saída).

Rodar:  python scripts/benchmark_router.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dealership_agent.config import settings  # noqa: E402
from dealership_agent.graph.router_jev import JevRouter  # noqa: E402
from dealership_agent.llm import _ROUTER_SYSTEM, get_llm_client  # noqa: E402
from dealership_agent.schemas.lead import LeadProfile  # noqa: E402

TEST_MESSAGES: list[str] = [
    "Vocês têm algum SUV disponível até 150 mil?",
    "Consigo financiar com 20 mil de entrada? Qual fica a parcela?",
    "Quero agendar um test-drive do Corolla Cross no sábado de manhã.",
]


def estimate_tokens(text: str) -> int:
    """Estimativa grosseira (~4 chars/token), só para comparar ordens de grandeza."""
    return max(1, len(text) // 4)


async def bench_classic(message: str) -> tuple[str, float, int]:
    llm = get_llm_client()
    started = time.perf_counter()
    decision = await llm.decide_route([HumanMessage(message)], LeadProfile())
    latency_ms = (time.perf_counter() - started) * 1000
    # Prompt generativo = system + mensagem; saída = JSON da decisão.
    tokens = estimate_tokens(_ROUTER_SYSTEM + message) + estimate_tokens(
        decision.model_dump_json()
    )
    return decision.intent.value, latency_ms, tokens


async def bench_jev(
    router: JevRouter, message: str
) -> tuple[str, float, int, float, str, float]:
    decision = await router.route_message(message)
    return (
        decision.intent.value,
        decision.latency_ms,
        decision.total_tokens,
        decision.probability,
        decision.source,
        decision.cost_usd,
    )


async def main() -> None:
    router = JevRouter()
    classic_mode = (
        "LLM real" if settings.llm_configured else "fallback offline (sem OPENAI_API_KEY)"
    )
    jev_mode = (
        "OpenRouter (typesafe/jev-1.14)"
        if router.configured
        else "fallback clássico (sem OPENROUTER_API_KEY)"
    )
    print(f"Roteador clássico: {classic_mode}")
    print(f"Roteador Jev:      {jev_mode}")
    print("=" * 78)

    tot_classic_ms = tot_jev_ms = 0.0
    tot_classic_tok = tot_jev_tok = 0
    for i, message in enumerate(TEST_MESSAGES, 1):
        c_intent, c_ms, c_tok = await bench_classic(message)
        j_intent, j_ms, j_tok, j_prob, j_src, j_cost = await bench_jev(router, message)
        tot_classic_ms += c_ms
        tot_jev_ms += j_ms
        tot_classic_tok += c_tok
        tot_jev_tok += j_tok
        print(f'\n[{i}] Lead: "{message}"')
        print(
            f"  Clássico (LLM generativo) : intent={c_intent:<20} "
            f"latência={c_ms:8.2f} ms  tokens≈{c_tok}"
        )
        print(
            f"  Jev (typesafe/jev-1.14)   : intent={j_intent:<20} latência={j_ms:8.2f} ms  "
            f"tokens={j_tok}  custo=US$ {j_cost:.7f}  p={j_prob:.2f}  fonte={j_src}"
        )

    print("\n" + "=" * 78)
    print(f"TOTAL clássico: latência={tot_classic_ms:.2f} ms  tokens≈{tot_classic_tok}")
    print(f"TOTAL Jev:      latência={tot_jev_ms:.2f} ms  tokens≈{tot_jev_tok}")
    if tot_jev_tok < tot_classic_tok:
        saved = 100 * (1 - tot_jev_tok / max(tot_classic_tok, 1))
        print(f"Economia estimada de tokens do Jev: {saved:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())

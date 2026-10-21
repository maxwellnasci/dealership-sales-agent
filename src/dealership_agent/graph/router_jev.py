"""Adapter de roteamento rápido Jev (TypeSafe AI) via OpenRouter.

Ports & Adapters: o grafo depende do contrato `LLMClient.decide_route`
(porta, em `llm.py`). Este módulo é um **adapter** alternativo só para a
decisão de rota: em vez de gerar texto com um LLM generativo, ele classifica
a mensagem do lead com o modelo `typesafe/jev-1.14` no endpoint de decisões
do OpenRouter (`/api/alpha/decisions` — modelos de decisão não aceitam
`/chat/completions`) e devolve a mesma `RouteDecision` tipada, acrescida da
probabilidade da classe.

Fallback (requisito de segurança): sem `OPENROUTER_API_KEY` — ou em qualquer
erro de rede/parse — a decisão é delegada ao roteador clássico injetado
(por padrão o `TemplateLLMClient` determinístico, sem rede). Ou seja, o
clássico continua sendo o comportamento padrão; o Jev só entra quando há
chave configurada.
"""

from __future__ import annotations

import time
from typing import Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from dealership_agent.config import secret, settings
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.routing import Intent, RouteDecision

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.14"

# Rótulos que o Jev devolve -> Intent do grafo. `handoff_human` é o nome do
# nó de destino; a Intent correspondente é `Intent.HANDOFF` ("handoff").
JEV_LABEL_TO_INTENT: dict[str, Intent] = {
    "search_inventory": Intent.SEARCH_INVENTORY,
    "financing": Intent.FINANCING,
    "schedule_test_drive": Intent.SCHEDULE_TEST_DRIVE,
    "handoff_human": Intent.HANDOFF,
    "handoff": Intent.HANDOFF,
    "qualify": Intent.QUALIFY,
    "small_talk": Intent.SMALL_TALK,
}

# Rótulos canônicos pedidos ao Jev (1:1 com os destinos do supervisor).
JEV_LABELS: tuple[str, ...] = (
    "qualify",
    "search_inventory",
    "financing",
    "schedule_test_drive",
    "handoff_human",
    "small_talk",
)

# Critérios da pergunta `choice`: rótulo -> descrição usada pelo Jev.
JEV_CRITERIA: dict[str, str] = {
    "qualify": "O lead descreve necessidade, gosto, orçamento ou responde perguntas de descoberta.",
    "search_inventory": "O lead procura veículos no estoque (modelo, categoria, preço, ano).",
    "financing": "O lead pergunta sobre financiamento, entrada, parcelas ou simulação.",
    "schedule_test_drive": "O lead quer agendar ou remarcar um test-drive.",
    "handoff_human": "O lead pede um vendedor humano, reclama ou quer negociar preço/desconto.",
    "small_talk": "Saudações iniciais ou assuntos fora do escopo de compra de veículo.",
}
JEV_INSTRUCTIONS = "Classifique a intenção da última mensagem do lead de uma concessionária."


class JevRouteDecision(BaseModel):
    """Decisão tipada do Jev: intenção + probabilidade da classe."""

    intent: Intent
    probability: float = Field(ge=0.0, le=1.0, description="Probabilidade da intenção escolhida.")
    label: str = Field(description="Rótulo bruto devolvido pelo classificador.")
    source: Literal["jev", "fallback-classico"] = "jev"
    latency_ms: float = 0.0
    # Uso reportado pelo OpenRouter (`usage.input_tokens/output_tokens/cost`).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_route_decision(self) -> RouteDecision:
        return RouteDecision(
            intent=self.intent,
            rationale=f"jev:{self.label} p={self.probability:.2f} ({self.source})",
        )


def _normalize_label(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def parse_jev_payload(payload: dict) -> tuple[str, float, int, int, float]:
    """Extrai (label, probabilidade, input_tokens, output_tokens, custo_usd).

    Formato do endpoint de decisões do OpenRouter::

        {"answers": {"intent": {"type": "choice", "choice": "financing",
                                "probabilities": {"financing": 0.97, ...},
                                "confidence": 0.97}},
         "usage": {"input_tokens": 374, "output_tokens": 55, "cost": 1.5e-05}}
    """
    usage = payload.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cost = float(usage.get("cost") or 0.0)

    answer = (payload.get("answers") or {}).get("intent")
    if not answer:
        raise ValueError("resposta do OpenRouter sem 'answers.intent'")
    raw_label = str(answer.get("choice") or "")
    label = _normalize_label(raw_label)

    probabilities = answer.get("probabilities") or {}
    probability = probabilities.get(raw_label, answer.get("confidence"))
    if probability is None:
        probability = 1.0 if label in JEV_LABEL_TO_INTENT else 0.0
    probability = max(0.0, min(1.0, float(probability)))
    return label, probability, input_tokens, output_tokens, cost


class JevRouter:
    """Roteador Jev com fallback automático para o roteador clássico.

    Uso direto (benchmark/scripts)::

        decision = await JevRouter().route_message("quero financiar um SUV")

    Uso como porta do grafo (mesma assinatura de `LLMClient.decide_route`)::

        decision = await JevRouter().decide_route(messages, lead)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = JEV_MODEL,
        url: str = OPENROUTER_DECISIONS_URL,
        fallback=None,
        timeout_s: float | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else secret(settings.openrouter_api_key)
        self._model = model
        self._url = url
        self._timeout_s = timeout_s if timeout_s is not None else settings.jev_timeout_s
        if fallback is None:
            from dealership_agent.llm import TemplateLLMClient

            fallback = TemplateLLMClient()
        self._fallback = fallback

    @property
    def configured(self) -> bool:
        """True quando há chave do OpenRouter (senão, só fallback clássico)."""
        return bool(self._api_key)

    async def route_message(self, message: str) -> JevRouteDecision:
        """Classifica uma mensagem solta do lead (atalho p/ benchmark)."""
        messages = [{"role": "user", "content": message}]
        return await self._route(messages, LeadProfile())

    async def decide_route(
        self, messages: list[BaseMessage], lead: LeadProfile
    ) -> RouteDecision:
        """Porta compatível com `LLMClient.decide_route` (supervisor)."""
        raw = [
            {"role": "user", "content": str(getattr(m, "content", "") or "")} for m in messages
        ]
        decision = await self._route(raw, lead)
        return decision.to_route_decision()

    async def _route(self, messages: list[dict], lead: LeadProfile) -> JevRouteDecision:
        started = time.perf_counter()
        if self.configured:
            try:
                decision = await self._classify(messages)
                decision.latency_ms = (time.perf_counter() - started) * 1000
                return decision
            except Exception:
                pass  # qualquer falha -> fallback clássico, nunca quebra o turno
        decision = await self._fallback_decision(messages, lead)
        decision.latency_ms = (time.perf_counter() - started) * 1000
        return decision

    async def _classify(self, messages: list[dict]) -> JevRouteDecision:
        import httpx

        state = "\n".join(f"Lead: {m.get('content') or ''}" for m in messages)
        body = {
            "model": self._model,
            "state": state,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": JEV_INSTRUCTIONS,
                    "criteria": {label: JEV_CRITERIA[label] for label in JEV_LABELS},
                }
            },
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "dealership-sales-agent",
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(self._url, json=body, headers=headers)
            resp.raise_for_status()
        label, probability, prompt_tokens, completion_tokens, cost = parse_jev_payload(
            resp.json()
        )
        if label not in JEV_LABEL_TO_INTENT:
            raise ValueError(f"rótulo desconhecido do Jev: {label!r}")
        return JevRouteDecision(
            intent=JEV_LABEL_TO_INTENT[label],
            probability=probability,
            label=label,
            source="jev",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )

    async def _fallback_decision(
        self, messages: list[dict], lead: LeadProfile
    ) -> JevRouteDecision:
        from langchain_core.messages import HumanMessage

        lc_messages = [HumanMessage(str(m.get("content") or "")) for m in messages]
        classic = await self._fallback.decide_route(lc_messages, lead)
        label = classic.intent.value
        if classic.intent is Intent.HANDOFF:
            label = "handoff_human"
        return JevRouteDecision(
            intent=classic.intent,
            probability=1.0,
            label=label,
            source="fallback-classico",
        )


def get_router(fallback=None) -> JevRouter:
    """Factory: Jev quando há `OPENROUTER_API_KEY`, senão fallback clássico.

    O chamador não precisa ramificar: sem chave, `JevRouter` delega 100%
    das decisões ao roteador clássico (comportamento padrão preservado).
    """
    return JevRouter(model=settings.jev_model, fallback=fallback)

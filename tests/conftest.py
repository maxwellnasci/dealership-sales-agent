"""Fixtures de teste: grafo com checkpointer em memória e LLMs determinísticos.

Nenhum teste toca a rede nem depende de API key. O `TemplateLLMClient`
(determinístico) cobre os fluxos; clientes especializados cobrem casos de
borda (alucinação, roteamento forçado, juiz).
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from dealership_agent.graph.builder import build_graph
from dealership_agent.graph.state import initial_state
from dealership_agent.llm import LLMClient, TemplateLLMClient
from dealership_agent.persistence.checkpointer import checkpoint_serde
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.routing import Intent, RouteDecision


class ScriptedLLMClient(TemplateLLMClient):
    """TemplateLLMClient com pontos de injeção para forçar comportamento.

    - `forced_intent`: se setado, `decide_route` sempre devolve essa intenção.
    - `compose_override`: se setado, `compose_reply` devolve esse texto fixo
      (usado para simular um LLM que tenta alucinar valores).
    """

    def __init__(
        self, forced_intent: Intent | None = None, compose_override: str | None = None
    ) -> None:
        self.forced_intent = forced_intent
        self.compose_override = compose_override

    async def decide_route(self, messages: list[BaseMessage], lead: LeadProfile) -> RouteDecision:
        if self.forced_intent is not None:
            return RouteDecision(intent=self.forced_intent, rationale="forced")
        return await super().decide_route(messages, lead)

    async def compose_reply(self, system_prompt: str, messages: list[BaseMessage]) -> str:
        if self.compose_override is not None:
            return self.compose_override
        return await super().compose_reply(system_prompt, messages)


@pytest.fixture
def make_graph():
    """Fábrica: recebe um LLMClient e devolve um grafo compilado novo (memória)."""

    def _factory(llm: LLMClient | None = None):
        return build_graph(InMemorySaver(serde=checkpoint_serde()), llm or TemplateLLMClient())

    return _factory


@pytest.fixture
def new_thread_config():
    """Config de invoke com um thread_id único (= telefone do lead)."""

    def _factory(phone: str | None = None) -> dict:
        return {"configurable": {"thread_id": phone or f"+55419{uuid.uuid4().hex[:8]}"}}

    return _factory


@pytest.fixture
def run_turn():
    """Helper: roda um turno e devolve o estado final."""

    async def _run(graph, text: str, config: dict) -> dict:
        return await graph.ainvoke(initial_state(text, config["configurable"]["thread_id"]), config)

    return _run

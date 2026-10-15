"""API FastAPI que simula o webhook de WhatsApp.

Uma mensagem do lead -> um turno do grafo, com `thread_id = telefone`. O
lifespan abre o checkpointer e compila o grafo uma vez; cada request reaproveita
a instância. Em produção, o `checkpoint_backend=postgres` dá memória durável
entre mensagens; aqui o endpoint é o ponto de integração com o provedor de
WhatsApp (Cloud API, Evolution, etc.).

Rodar:  uvicorn dealership_agent.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.types import Command
from pydantic import BaseModel

from dealership_agent.graph.builder import build_graph
from dealership_agent.graph.state import initial_state
from dealership_agent.llm import get_llm_client
from dealership_agent.observability.tracing import get_langfuse_handler
from dealership_agent.persistence.checkpointer import checkpointer_context


class InboundMessage(BaseModel):
    phone: str
    text: str


class ResumeMessage(BaseModel):
    phone: str
    rep_text: str  # mensagem do vendedor humano para retomar um handoff


class AgentReply(BaseModel):
    reply: str | None
    interrupted: bool = False
    interrupt_payload: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with checkpointer_context() as checkpointer:
        app.state.graph = build_graph(checkpointer, get_llm_client())
        yield


app = FastAPI(title="Dealership Sales Agent", lifespan=lifespan)


def _run_config(phone: str) -> dict:
    handler = get_langfuse_handler()
    config: dict = {"configurable": {"thread_id": phone}}
    if handler:
        config["callbacks"] = [handler]
    return config


def _to_reply(result: dict) -> AgentReply:
    interrupts = result.get("__interrupt__")
    if interrupts:
        payload = getattr(interrupts[0], "value", None)
        return AgentReply(reply=None, interrupted=True, interrupt_payload=payload)
    return AgentReply(reply=result.get("final_reply"))


@app.post("/webhook", response_model=AgentReply)
async def webhook(msg: InboundMessage) -> AgentReply:
    """Recebe uma mensagem do lead e devolve a resposta do agente.

    Se o grafo pausar num handoff (`interrupt`), devolve o payload para o
    operador e fica aguardando um POST em /resume.
    """
    result = await app.state.graph.ainvoke(
        initial_state(msg.text, msg.phone), config=_run_config(msg.phone)
    )
    return _to_reply(result)


@app.post("/resume", response_model=AgentReply)
async def resume(msg: ResumeMessage) -> AgentReply:
    """Retoma um turno suspenso num handoff, com a resposta do vendedor humano."""
    result = await app.state.graph.ainvoke(
        Command(resume=msg.rep_text), config=_run_config(msg.phone)
    )
    return _to_reply(result)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

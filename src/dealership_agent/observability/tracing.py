"""Observabilidade com Langfuse (opcional, no-op gracioso).

Mede custo por conversa e permite anexar scores de qualidade (LLM-as-judge) a
cada trace. Se as chaves do Langfuse não estiverem configuradas, tudo vira
no-op silencioso — a demo e os testes nunca quebram por falta de Langfuse.

API do Langfuse v3+: o cliente `Langfuse` é registrado uma vez com as chaves;
o `CallbackHandler` (em `langfuse.langchain`) o resolve pela `public_key`.

Uso:
    handler = get_langfuse_handler()        # -> CallbackHandler | None
    config = {"callbacks": [handler]} if handler else {}
    await graph.ainvoke(state, config={**config, "configurable": {...}})
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from dealership_agent.config import secret, settings


@lru_cache(maxsize=1)
def _langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and secret(settings.langfuse_secret_key))


@lru_cache(maxsize=1)
def _client() -> Any:
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=secret(settings.langfuse_secret_key),
        host=settings.langfuse_host,
    )


def get_langfuse_handler() -> Any | None:
    """Callback handler do LangChain para Langfuse, ou None se desabilitado."""
    if not _langfuse_enabled():
        return None
    try:
        from langfuse.langchain import CallbackHandler

        _client()  # registra as credenciais antes do handler procurar o cliente
        return CallbackHandler(public_key=settings.langfuse_public_key)
    except Exception:
        # Dependência ausente ou versão incompatível: degrada para no-op.
        return None


def score_trace(trace_id: str, name: str, value: float, comment: str = "") -> None:
    """Anexa um score (ex: nota do LLM-as-judge) a um trace. No-op se desabilitado."""
    if not _langfuse_enabled():
        return
    try:
        _client().create_score(trace_id=trace_id, name=name, value=value, comment=comment)
    except Exception:
        return

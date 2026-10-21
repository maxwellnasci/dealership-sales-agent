"""Observabilidade Langfuse: no-op sem chave, credenciais em claro só no SDK."""

from __future__ import annotations

import sys
import types

import pytest
from pydantic import SecretStr

from dealership_agent.config import settings
from dealership_agent.observability import tracing


@pytest.fixture(autouse=True)
def _reset_caches():
    tracing._langfuse_enabled.cache_clear()
    tracing._client.cache_clear()
    yield
    tracing._langfuse_enabled.cache_clear()
    tracing._client.cache_clear()


@pytest.fixture
def fake_langfuse(monkeypatch):
    """Substitui o SDK por fakes que registram as chamadas (sem rede)."""
    calls: dict = {"scores": []}

    class Langfuse:
        def __init__(self, **kwargs):
            calls["client"] = kwargs

        def create_score(self, **kwargs):
            calls["scores"].append(kwargs)

    class CallbackHandler:
        def __init__(self, **kwargs):
            calls["handler"] = kwargs

    root = types.ModuleType("langfuse")
    root.Langfuse = Langfuse
    lc = types.ModuleType("langfuse.langchain")
    lc.CallbackHandler = CallbackHandler
    monkeypatch.setitem(sys.modules, "langfuse", root)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", lc)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-1")
    monkeypatch.setattr(settings, "langfuse_secret_key", SecretStr("sk-lf-1"))
    monkeypatch.setattr(settings, "langfuse_host", "https://lf.local")
    return calls


def test_sem_chaves_tudo_e_no_op(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", None)
    monkeypatch.setattr(settings, "langfuse_secret_key", None)
    assert tracing.get_langfuse_handler() is None
    tracing.score_trace("t1", "judge", 0.9)  # não levanta


def test_secret_vazio_conta_como_desabilitado(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "pk")
    monkeypatch.setattr(settings, "langfuse_secret_key", SecretStr(""))
    assert tracing.get_langfuse_handler() is None


def test_handler_registra_cliente_com_credenciais(fake_langfuse):
    handler = tracing.get_langfuse_handler()
    assert handler is not None
    assert fake_langfuse["client"] == {
        "public_key": "pk-lf-1", "secret_key": "sk-lf-1", "host": "https://lf.local",
    }
    assert fake_langfuse["handler"] == {"public_key": "pk-lf-1"}


def test_score_trace_envia_score(fake_langfuse):
    tracing.score_trace("t1", "judge", 0.8, comment="ok")
    assert fake_langfuse["scores"] == [
        {"trace_id": "t1", "name": "judge", "value": 0.8, "comment": "ok"}
    ]


def test_falha_do_sdk_degrada_para_no_op(fake_langfuse, monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("sdk quebrado")

    monkeypatch.setattr(sys.modules["langfuse"], "Langfuse", _boom)
    assert tracing.get_langfuse_handler() is None
    tracing.score_trace("t1", "judge", 0.8)  # não levanta


def test_sdk_instalado_expoe_a_api_usada():
    # Regressão: `langfuse.callback` sumiu no v3 e o handler virava None
    # em silêncio. Trava os caminhos de import contra o SDK real.
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler  # noqa: F401

    assert hasattr(Langfuse, "create_score")

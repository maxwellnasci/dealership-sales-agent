"""Credenciais como SecretStr e adaptador OpenAI com timeout explícito.

O `ChatOpenAI` é substituído por um fake: nada toca a rede.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr

from dealership_agent import llm as llm_module
from dealership_agent.config import Settings, secret, settings
from dealership_agent.graph import router_jev
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.routing import Intent, RouteDecision

# --- SecretStr ---------------------------------------------------------------


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_credenciais_nao_vazam_em_repr_nem_dump():
    s = _settings(
        openai_api_key="sk-live-123",
        openrouter_api_key="or-456",
        langfuse_secret_key="lf-789",
        postgres_dsn="postgresql://u:pw-000@db/x",
    )
    dumped = f"{s!r} {s} {s.model_dump()}"
    for raw in ("sk-live-123", "or-456", "lf-789", "pw-000"):
        assert raw not in dumped
    assert isinstance(s.openai_api_key, SecretStr)
    assert secret(s.openai_api_key) == "sk-live-123"
    assert secret(s.postgres_dsn) == "postgresql://u:pw-000@db/x"


def test_flags_de_configuracao_respeitam_chave_vazia():
    assert _settings().llm_configured is False
    assert _settings(openai_api_key="").llm_configured is False  # `OPENAI_API_KEY=` no .env
    assert _settings(openai_api_key="sk").llm_configured is True
    assert _settings(openrouter_api_key="").jev_configured is False
    assert _settings(openrouter_api_key="or").jev_configured is True


def test_credencial_lida_do_ambiente(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    s = _settings()
    assert secret(s.openai_api_key) == "sk-env"
    assert "sk-env" not in repr(s)


def test_secret_helper():
    assert secret(None) is None
    assert secret(SecretStr("")) is None
    assert secret(SecretStr("x")) == "x"


def test_jev_router_le_chave_e_timeout_do_settings(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr("or-key"))
    monkeypatch.setattr(settings, "jev_timeout_s", 2.5)
    router = router_jev.JevRouter()
    assert router._api_key == "or-key"
    assert router._timeout_s == 2.5
    assert router_jev.JevRouter(timeout_s=1.0)._timeout_s == 1.0


# --- OpenAILLMClient ---------------------------------------------------------


class _FakeStructured:
    def __init__(self, model, schema):
        self._model, self._schema = model, schema

    async def ainvoke(self, prompt):
        self._model.prompts.append(prompt)
        result = self._model.structured[self._schema]
        if isinstance(result, Exception):
            raise result
        return result


class _FakeChatOpenAI:
    instances: list[_FakeChatOpenAI] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prompts: list = []
        self.structured: dict = {}
        self.reply: object = AIMessage(content="olá!")
        _FakeChatOpenAI.instances.append(self)

    def with_structured_output(self, schema):
        return _FakeStructured(self, schema)

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture
def fake_openai(monkeypatch):
    import langchain_openai

    _FakeChatOpenAI.instances.clear()
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-test"))
    monkeypatch.setattr(settings, "openai_base_url", "https://llm.local/v1")
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    monkeypatch.setattr(settings, "llm_timeout_s", 12.0)
    monkeypatch.setattr(settings, "llm_max_retries", 1)
    client = llm_module.OpenAILLMClient()
    return client, _FakeChatOpenAI.instances[-1]


def test_cliente_openai_tem_timeout_e_retries_explicitos(fake_openai):
    _, model = fake_openai
    assert model.kwargs["timeout"] == 12.0
    assert model.kwargs["max_retries"] == 1
    assert model.kwargs["api_key"] == "sk-test"  # em claro só aqui, no SDK
    assert model.kwargs["base_url"] == "https://llm.local/v1"


def test_factory_escolhe_cliente_real_com_chave(fake_openai, monkeypatch):
    assert isinstance(llm_module.get_llm_client(), llm_module.OpenAILLMClient)
    monkeypatch.setattr(settings, "openai_api_key", None)
    assert isinstance(llm_module.get_llm_client(), llm_module.TemplateLLMClient)


async def test_roteador_classico_usa_structured_output_com_contexto(fake_openai):
    client, model = fake_openai
    model.structured[RouteDecision] = RouteDecision(intent=Intent.FINANCING, rationale="x")
    lead = LeadProfile(body_type="suv")
    decision = await client.decide_route([HumanMessage("e o financiamento?")], lead)
    assert decision.intent is Intent.FINANCING
    assert "body_type=suv" in model.prompts[-1][0].content


async def test_roteador_classico_cai_em_qualify_quando_llm_falha(fake_openai):
    client, model = fake_openai
    model.structured[RouteDecision] = TimeoutError("timeout")
    decision = await client.decide_route([HumanMessage("oi")], LeadProfile())
    assert decision.intent is Intent.QUALIFY


async def test_com_chave_jev_roteia_pelo_jev_com_fallback_classico(fake_openai, monkeypatch):
    client, model = fake_openai
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr("or"))
    model.structured[RouteDecision] = RouteDecision(intent=Intent.HANDOFF, rationale="x")

    async def _boom(self, messages):
        raise TimeoutError("jev fora")

    monkeypatch.setattr(router_jev.JevRouter, "_classify", _boom)
    decision = await client.decide_route([HumanMessage("quero um vendedor")], LeadProfile())
    assert decision.intent is Intent.HANDOFF
    assert "fallback-classico" in decision.rationale


async def test_extract_lead(fake_openai):
    client, model = fake_openai
    model.structured[LeadProfile] = LeadProfile(desired_model="Corolla")
    assert (await client.extract_lead("quero corolla", LeadProfile())).desired_model == "Corolla"
    model.structured[LeadProfile] = None
    assert await client.extract_lead("oi", LeadProfile()) == LeadProfile()
    model.structured[LeadProfile] = TimeoutError("timeout")
    assert await client.extract_lead("oi", LeadProfile()) == LeadProfile()


async def test_compose_reply(fake_openai):
    client, model = fake_openai
    assert await client.compose_reply("sys", [HumanMessage("oi")]) == "olá!"
    model.reply = "texto cru"
    assert await client.compose_reply("sys", []) == "texto cru"


@pytest.mark.parametrize(
    ("content", "expected"), [("0.8", 0.8), ("1.7 ótimo", 1.0), ("-2", 0.0), ("nota boa", 0.0)]
)
async def test_judge_normaliza_nota(fake_openai, content, expected):
    client, model = fake_openai
    model.reply = AIMessage(content=content)
    assert await client.judge("rubrica", "conversa") == expected

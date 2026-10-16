"""Porta (interface) e adaptador do modelo de linguagem.

Padrão ports & adapters: o grafo depende de `LLMClient` (abstrato), não de uma
SDK concreta. Em produção/demo usamos `OpenAILLMClient` (LangChain +
`with_structured_output`/`bind_tools`). Nos testes usamos um cliente scriptado
e determinístico (tests/conftest.py) — pytest roda rápido, sem custo e sem rede.

Assim o caminho de produção mantém o uso idiomático de structured output e
tool calling do LangChain, sem acoplar a suíte de testes a um LLM
não-determinístico.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from dealership_agent.config import settings
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.schemas.routing import Intent, RouteDecision


class LLMClient(ABC):
    """Contrato mínimo de LLM de que o grafo precisa."""

    @abstractmethod
    async def decide_route(self, messages: list[BaseMessage], lead: LeadProfile) -> RouteDecision:
        """Classifica a última mensagem do lead numa `Intent` (structured output)."""

    @abstractmethod
    async def extract_lead(self, message: str, current: LeadProfile) -> LeadProfile:
        """Extrai campos do perfil a partir da mensagem (structured output)."""

    @abstractmethod
    async def compose_reply(self, system_prompt: str, messages: list[BaseMessage]) -> str:
        """Gera a resposta em linguagem natural a partir do contexto já montado."""

    @abstractmethod
    async def judge(self, rubric: str, conversation: str) -> float:
        """LLM-as-judge: pontua a qualidade de 0.0 a 1.0 (usado em avaliação)."""


_ROUTER_SYSTEM = """Você é o roteador de um agente de vendas de uma revenda de veículos.
Classifique a ÚLTIMA mensagem do cliente em uma intenção. Regras:
- qualify: cliente descreve necessidade, gosto, orçamento ou responde perguntas de descoberta.
- search_inventory: cliente quer ver carros disponíveis / pede opções / pergunta o que tem.
- schedule_test_drive: cliente quer agendar, marcar ou fazer um test-drive.
- financing: cliente pergunta sobre parcelas, entrada, financiamento ou condições de pagamento.
- handoff: cliente pede explicitamente falar com um vendedor/humano, reclama ou quer negociar preço.
- small_talk: saudações ou assuntos fora do escopo de compra de veículo.
Escolha a intenção mais específica aplicável."""


def _lead_summary(lead: LeadProfile) -> str:
    """Resumo compacto do que já se sabe do lead, para dar contexto ao LLM.

    Deixa o roteador e o extrator cientes do histórico: ex., uma mensagem curta
    como "e o financiamento?" é ambígua sozinha, mas clara sabendo que o lead
    já escolheu um SUV. Também ajuda o extrator a só complementar o perfil.
    """
    known = {k: v for k, v in lead.model_dump().items() if v is not None and k != "phone"}
    if not known:
        return "nenhuma informação coletada ainda."
    return ", ".join(f"{k}={v}" for k, v in known.items())


class OpenAILLMClient(LLMClient):
    """Adaptador real sobre um endpoint OpenAI-compatível (via LangChain)."""

    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI

        kwargs: dict = {"model": settings.llm_model, "temperature": settings.llm_temperature}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._model = ChatOpenAI(**kwargs)

    async def decide_route(self, messages: list[BaseMessage], lead: LeadProfile) -> RouteDecision:
        structured = self._model.with_structured_output(RouteDecision)
        router_sys = f"{_ROUTER_SYSTEM}\n\nPerfil já conhecido do lead: {_lead_summary(lead)}"
        prompt: list[BaseMessage] = [SystemMessage(router_sys), *messages]
        try:
            return await structured.ainvoke(prompt)
        except Exception:
            # Fallback seguro: na dúvida, continuamos a descoberta.
            return RouteDecision(intent=Intent.QUALIFY, rationale="fallback: erro no roteador")

    async def extract_lead(self, message: str, current: LeadProfile) -> LeadProfile:
        structured = self._model.with_structured_output(LeadProfile)
        prompt = [
            SystemMessage(
                "Extraia APENAS o que o cliente afirmou nesta mensagem sobre o carro que "
                "procura. Não invente nada; deixe campos ausentes como null. Atenção: "
                "`name` é o PRIMEIRO NOME DA PESSOA (ex: 'Ana'), nunca um modelo de carro. "
                "Marca do carro vai em `desired_make` (ex: Toyota) e o modelo em "
                "`desired_model` (ex: 'Corolla' -> desired_model='Corolla').\n"
                "Contexto já coletado (use só para desambiguar, NÃO repita): "
                f"{_lead_summary(current)}"
            ),
            HumanMessage(message),
        ]
        try:
            extracted = await structured.ainvoke(prompt)
            return extracted or LeadProfile()
        except Exception:
            return LeadProfile()

    async def compose_reply(self, system_prompt: str, messages: list[BaseMessage]) -> str:
        prompt: list[BaseMessage] = [SystemMessage(system_prompt), *messages]
        result = await self._model.ainvoke(prompt)
        if isinstance(result, AIMessage):
            return str(result.content)
        return str(result)

    async def judge(self, rubric: str, conversation: str) -> float:
        prompt = [
            SystemMessage(
                "Você é um avaliador rigoroso. Dada a rubrica e a conversa, responda APENAS "
                "um número entre 0 e 1 (ex: 0.8) representando o quão bem o agente atendeu."
            ),
            HumanMessage(f"RUBRICA:\n{rubric}\n\nCONVERSA:\n{conversation}"),
        ]
        result = await self._model.ainvoke(prompt)
        try:
            return max(0.0, min(1.0, float(str(result.content).strip().split()[0])))
        except (ValueError, IndexError):
            return 0.0


_MAKES = [
    "toyota", "honda", "jeep", "volkswagen", "vw", "chevrolet", "hyundai", "fiat",
]
_MODELS = [
    "corolla cross", "corolla", "civic", "hr-v", "hrv", "renegade", "compass", "nivus",
    "polo", "t-cross", "tcross", "onix", "tracker", "hb20", "creta", "toro", "pulse",
]
_BODY_TYPES = {
    "sedan": "sedan", "suv": "suv", "hatch": "hatch",
    "picape": "picape", "caminhonete": "picape",
}


class TemplateLLMClient(LLMClient):
    """Fallback determinístico, SEM rede, para quando não há API key.

    Mantém a demo rodando offline (roteamento por palavra-chave, extração por
    vocabulário, resposta = o próprio rascunho factual já montado pelo nó de
    síntese). Não é tão fluente quanto um LLM real, mas é honesto e nunca
    alucina. É o "fallback gracioso" do `demo.py`.
    """

    @staticmethod
    def _last_text(messages: list[BaseMessage]) -> str:
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None
            )
            if content:
                return str(content).lower()
        return ""

    async def decide_route(self, messages: list[BaseMessage], lead: LeadProfile) -> RouteDecision:
        text = self._last_text(messages)
        if any(k in text for k in ("vendedor", "humano", "atendente", "desconto", "negociar")):
            return RouteDecision(intent=Intent.HANDOFF, rationale="keyword: humano/negociação")
        if any(k in text for k in ("test-drive", "test drive", "testar", "agendar", "marcar")):
            return RouteDecision(intent=Intent.SCHEDULE_TEST_DRIVE, rationale="keyword: agendar")
        if any(k in text for k in ("financia", "parcela", "entrada", "parcelar", "a prazo")):
            return RouteDecision(intent=Intent.FINANCING, rationale="keyword: financiamento")
        if any(k in text for k in ("tem", "disponí", "opç", "mostra", "ver carro", "quais")):
            return RouteDecision(intent=Intent.SEARCH_INVENTORY, rationale="keyword: estoque")
        if any(k in text for k in _MAKES + _MODELS) or any(b in text for b in _BODY_TYPES):
            return RouteDecision(intent=Intent.QUALIFY, rationale="menciona carro")
        if any(k in text for k in ("mil", "r$", "orçamento", "até ", "família", "trabalho")):
            return RouteDecision(intent=Intent.QUALIFY, rationale="menciona necessidade")
        return RouteDecision(intent=Intent.SMALL_TALK, rationale="sem gatilho de compra")

    async def extract_lead(self, message: str, current: LeadProfile) -> LeadProfile:
        import re

        text = message.lower()
        data: dict = {}
        for model in _MODELS:
            if model in text:
                data["desired_model"] = model.title()
                break
        for make in _MAKES:
            if make in text:
                data["desired_make"] = "Volkswagen" if make == "vw" else make.title()
                break
        for key, norm in _BODY_TYPES.items():
            if key in text:
                data["body_type"] = norm
                break
        # Orçamento: "até 140 mil", "140.000", "R$ 140000"
        m = re.search(r"(\d{2,3})\s*mil", text)
        if m:
            data["budget_max"] = float(m.group(1)) * 1000
        else:
            m = re.search(r"r?\$?\s*(\d{2,3}(?:[.\s]?\d{3}))", text)
            if m:
                data["budget_max"] = float(m.group(1).replace(".", "").replace(" ", ""))
        if "família" in text or "familia" in text:
            data["purpose"] = "família"
        return LeadProfile(**data)

    async def compose_reply(self, system_prompt: str, messages: list[BaseMessage]) -> str:
        # O nó de síntese embute o rascunho factual em "RASCUNHO:\n...".
        marker = "RASCUNHO:\n"
        if marker in system_prompt:
            return system_prompt.split(marker, 1)[1].strip()
        return "Posso te ajudar a escolher um carro, simular financiamento ou agendar test-drive."

    async def judge(self, rubric: str, conversation: str) -> float:
        # Heurística offline: pontua por sinais de bom atendimento no texto.
        text = conversation.lower()
        score = 0.5
        if any(k in text for k in ("r$", "parcela", "test-drive", "agend")):
            score += 0.3
        if "?" in conversation:  # fez uma pergunta de avanço
            score += 0.2
        return min(1.0, score)


def get_llm_client() -> LLMClient:
    """Factory: usa o LLM real se houver credencial, senão o fallback offline.

    Em testes, injeta-se um cliente scriptado diretamente via `build_graph`.
    """
    if settings.llm_configured:
        return OpenAILLMClient()
    return TemplateLLMClient()

"""Tool de estoque (mock auto-contido).

Lê um catálogo de veículos de `data/inventory.json`. A chamada é envolvida por
`tenacity` para demonstrar **tool calling com retry**: falhas transitórias de
I/O (um `OSError` ao ler o arquivo, análogo a um timeout de CRM real) são
re-tentadas com backoff exponencial; um estoque simplesmente vazio NÃO é erro
e não dispara retry.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from dealership_agent.schemas.tools import InventoryQuery, Vehicle

_DATA_FILE = Path(__file__).resolve().parents[3] / "data" / "inventory.json"


class InventoryUnavailable(RuntimeError):
    """Falha transitória ao acessar o estoque (deve ser re-tentada)."""


@lru_cache(maxsize=1)
def _load_catalog() -> list[Vehicle]:
    try:
        raw = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - exercitado via injeção no teste
        raise InventoryUnavailable(str(exc)) from exc
    return [Vehicle(**item) for item in raw]


def load_catalog() -> list[Vehicle]:
    """Catálogo completo (cacheado). Usado pelo guardrail para reconhecer modelos."""
    return _load_catalog()


@retry(
    retry=retry_if_exception_type(InventoryUnavailable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.05, max=0.5),
    reraise=True,
)
def search_inventory(query: InventoryQuery, limit: int = 3) -> list[Vehicle]:
    """Busca veículos que casem com a query. Determinístico e ordenado por preço.

    O retry cobre apenas `InventoryUnavailable` (falha de acesso), nunca um
    resultado vazio — distinção que separa erro de infraestrutura de resposta
    legítima de negócio.
    """
    catalog = _load_catalog()

    def matches(v: Vehicle) -> bool:
        if query.make and query.make.lower() not in v.make.lower():
            return False
        if query.model and query.model.lower() not in v.model.lower():
            return False
        if query.body_type and query.body_type.lower() != v.body_type.lower():
            return False
        if query.max_price is not None and v.price > query.max_price:
            return False
        return True

    hits = sorted((v for v in catalog if matches(v)), key=lambda v: v.price)
    return hits[:limit]

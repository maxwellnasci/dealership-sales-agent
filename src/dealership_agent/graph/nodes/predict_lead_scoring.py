"""Nó de scoring preditivo (mxos_ldm): roda logo após a qualificação.

Se o lead tem carro na troca (`ldm_vehicle_input`) e o grafo foi montado com
um `LDMPredictor`, calcula probabilidade de fechamento, giro e teto de
avaliação, e gera o Card Executivo do vendedor das 08h30. Sem carro ou sem
predictor, não faz nada.

Os números ficam em `ldm_prediction`/`seller_briefing_card` e NUNCA vão para a
resposta do lead — são informação interna do vendedor.

Fail-closed: FIPE nula/inválida -> `ldm_prediction=None` e card BLOQUEADO.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from dealership_agent.graph.state import GraphState
from dealership_agent.schemas.lead import LeadProfile
from dealership_agent.tools.financing import DEFAULT_DOWN_PAYMENT_PCT
from mxos_ldm.model import LDMPredictor
from mxos_ldm.sales_bridge import build_seller_card
from mxos_ldm.schemas import FipeIndisponivelError


def _interesse(lead: LeadProfile) -> str | None:
    label = " ".join(p for p in (lead.desired_make, lead.desired_model) if p)
    return label or None


async def predict_lead_scoring_node(
    state: GraphState, *, predictor: LDMPredictor | None
) -> dict:
    vehicle = state.get("ldm_vehicle_input")
    if predictor is None or vehicle is None:
        return {}

    lead = state.get("lead_profile") or LeadProfile()
    vehicle_data = vehicle.model_dump() if isinstance(vehicle, BaseModel) else dict(vehicle)
    # A entrada ainda não é coletada na conversa: usa a premissa da simulação
    # de financiamento da revenda (mesma regra da tool).
    ldm_lead = {
        "lead_id": lead.phone or "desconhecido",
        "nome": lead.name,
        "valor_entrada_pct": DEFAULT_DOWN_PAYMENT_PCT,
        "veiculo_interesse": _interesse(lead),
    }

    try:
        prediction = predictor.predict_raw(ldm_lead, vehicle_data)
    except (FipeIndisponivelError, ValidationError):
        prediction = None

    card = build_seller_card(
        {
            "lead_profile": lead,
            "ldm_lead": ldm_lead,
            "ldm_vehicle": vehicle_data,
            "ldm_prediction": prediction,
        }
    )
    return {"ldm_prediction": prediction, "seller_briefing_card": card}

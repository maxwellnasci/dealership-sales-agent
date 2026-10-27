"""mxos_ldm — Large Data Model tabular automotivo da MXOS.

Módulo preditivo que enriquece o StateGraph do dealership-sales-agent com três
números determinísticos por lead com carro na troca:

  - probabilidade_fechamento (0-100)
  - giro_estimado_dias (tempo de pátio previsto do carro da troca)
  - avaliacao_fipe_recomendada (teto em % da FIPE para margem líquida alvo)

Fail-closed: sem FIPE válida, nada é previsto — o módulo levanta
`FipeIndisponivelError` em vez de inventar uma avaliação.
"""

from mxos_ldm.schemas import (
    FipeIndisponivelError,
    LeadData,
    PredictionOutput,
    VehicleAppraisalInput,
)

__all__ = [
    "FipeIndisponivelError",
    "LeadData",
    "PredictionOutput",
    "VehicleAppraisalInput",
]

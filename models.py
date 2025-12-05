# act_ie/models.py

from dataclasses import dataclass
from typing import List, Optional, Literal, Dict


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class NumericContext:
    """
    Minimal numeric context for v0.
    Extend this as needed (fertiliser prices, emissions, etc.).
    """
    milk_price_eur_per_litre: float
    milk_price_volatility_12m: float
    rainfall_anomaly_last_6m_mm: float
    ndvi_mean_last_season: float
    herd_size: int


RiskLevel = Literal["minimal", "limited", "high"]


@dataclass
class RiskAssessment:
    level: RiskLevel
    reason: str


@dataclass
class AnswerMeta:
    county: str
    sector: str
    retrieved_doc_ids: List[str]
    risk_assessment: RiskAssessment
    numeric_summary: str


@dataclass
class EngineAnswer:
    answer: str
    meta: AnswerMeta


# ---- Simple DTOs for API layer (FastAPI schemas will reuse these structures) ----

@dataclass
class AnswerRequest:
    query: str
    county: str
    sector: str
    numeric: Dict[str, float]  # keys align with NumericContext fields

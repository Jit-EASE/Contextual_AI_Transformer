from pydantic import BaseModel
from typing import List, Optional


class Document(BaseModel):
    doc_id: str
    title: str
    text: str


class NumericContext(BaseModel):
    milk_price_eur_per_litre: float
    milk_price_volatility_12m: float
    rainfall_anomaly_last_6m_mm: float
    ndvi_mean_last_season: float
    herd_size: int


class AnswerMeta(BaseModel):
    county: str
    sector: str
    retrieved_doc_ids: List[str]
    risk_assessment: str
    numeric_summary: str


class EngineAnswer(BaseModel):
    answer: str
    meta: AnswerMeta


# 🔥 ADD THIS MISSING CLASS (required by api_main.py)
class QueryRequest(BaseModel):
    query: str
    county: str
    sector: str
    numeric: NumericContext

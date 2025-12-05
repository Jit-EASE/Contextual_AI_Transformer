# act_ie/api_main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

from .engine import ActIEEngine
from .models import NumericContext


app = FastAPI(
    title="ACT-IE: Agri-Context Transformer for Ireland",
    version="0.1.0",
    description="Contextual AI backend for Irish agri-food analytics.",
)

# Initialise engine once (warm start for Railway)
engine = ActIEEngine()


class NumericContextInput(BaseModel):
    milk_price_eur_per_litre: float = Field(..., example=0.42)
    milk_price_volatility_12m: float = Field(..., example=0.03)
    rainfall_anomaly_last_6m_mm: float = Field(..., example=-75.0)
    ndvi_mean_last_season: float = Field(..., example=0.61)
    herd_size: int = Field(..., example=85)


class AnswerRequestBody(BaseModel):
    query: str = Field(..., example="What are the main short-term risks for grass-based dairy farmers?")
    county: str = Field(..., example="Cork")
    sector: str = Field(..., example="Dairy")
    numeric: NumericContextInput


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "message": "ACT-IE backend running"}


@app.post("/answer")
def answer(req: AnswerRequestBody) -> Dict[str, Any]:
    try:
        numeric = NumericContext(
            milk_price_eur_per_litre=req.numeric.milk_price_eur_per_litre,
            milk_price_volatility_12m=req.numeric.milk_price_volatility_12m,
            rainfall_anomaly_last_6m_mm=req.numeric.rainfall_anomaly_last_6m_mm,
            ndvi_mean_last_season=req.numeric.ndvi_mean_last_season,
            herd_size=req.numeric.herd_size,
        )

        result = engine.answer(
            query=req.query,
            numeric=numeric,
            county=req.county,
            sector=req.sector,
        )

        return {
            "answer": result.answer,
            "meta": {
                "county": result.meta.county,
                "sector": result.meta.sector,
                "retrieved_doc_ids": result.meta.retrieved_doc_ids,
                "risk": {
                    "level": result.meta.risk_assessment.level,
                    "reason": result.meta.risk_assessment.reason,
                },
                "numeric_summary": result.meta.numeric_summary,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

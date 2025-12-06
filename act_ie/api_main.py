# act_ie/api_main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .engine import ActIEEngine
from .numeric_context import NumericContext

app = FastAPI(
    title="ACT-IE API",
    description="Agri-Context Transformer for Ireland (Lazy-loaded version)",
    version="1.0"
)

# ---------------------------------------------
# Lazy-loaded ACT-IE engine (fixes Railway cold start)
# ---------------------------------------------

engine = None

def get_engine():
    global engine
    if engine is None:
        try:
            engine = ActIEEngine()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Engine initialization failed: {e}")
    return engine


# ---------------------------------------------
# Request Model
# ---------------------------------------------

class QueryRequest(BaseModel):
    query: str
    county: str
    sector: str

    milk_price_eur_per_litre: float
    milk_price_volatility_12m: float
    rainfall_anomaly_last_6m_mm: float
    ndvi_mean_last_season: float
    herd_size: int


# ---------------------------------------------
# Endpoints
# ---------------------------------------------

@app.get("/")
def root():
    return {
        "message": "ACT-IE API is running. Use /docs to test the endpoints."
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/act-ie/query")
def act_ie_query(payload: QueryRequest):

    engine = get_engine()  # Lazy initialization

    numeric = NumericContext(
        milk_price_eur_per_litre=payload.milk_price_eur_per_litre,
        milk_price_volatility_12m=payload.milk_price_volatility_12m,
        rainfall_anomaly_last_6m_mm=payload.rainfall_anomaly_last_6m_mm,
        ndvi_mean_last_season=payload.ndvi_mean_last_season,
        herd_size=payload.herd_size,
    )

    try:
        response = engine.answer(
            query=payload.query,
            numeric=numeric,
            county=payload.county,
            sector=payload.sector,
            top_k_docs=5,
            max_tokens=450
        )
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ACT-IE error: {e}")

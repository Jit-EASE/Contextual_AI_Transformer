# act_ie/api_main.py

import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .engine import ActIEEngine
from .models import QueryRequest

app = FastAPI(title="ACT-IE API")

# DO NOT INIT ANYTHING HEAVY HERE
engine = None


@app.on_event("startup")
async def startup_event():
    global engine
    try:
        engine = ActIEEngine()   # Safe version, lazy inside
    except Exception as e:
        print("ENGINE INIT ERROR:", e)
        traceback.print_exc()


@app.get("/")
async def root():
    """
    Debug endpoint to dump the REAL startup error.
    """
    try:
        return {"status": "ok", "message": "ACT-IE API root reachable."}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )


@app.post("/initialize")
async def initialize():
    global engine
    try:
        if engine is None:
            return {"error": "Engine not initialized"}
        engine.build_embeddings()
        return {"status": "embeddings_ready"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )


@app.post("/answer")
async def answer(request: QueryRequest):
    try:
        if engine is None:
            return {"error": "Engine not initialized"}
        if not engine.embeddings_ready:
            return {"error": "Embeddings not initialized"}
        response = engine.answer(
            query=request.query,
            numeric=request.numeric,
            county=request.county,
            sector=request.sector,
        )
        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )

# act_ie/config.py

import os
from typing import List


class Config:
    # ----- Document / corpus paths -----
    CORPUS_CSV_PATH: str = os.getenv(
        "ACT_IE_CORPUS_CSV_PATH",
        "/Users/jit/Downloads/agri_corpus.csv"
    )

    # ----- Embeddings (OpenAI) -----
    EMBEDDING_MODEL: str = os.getenv("ACT_IE_EMBEDDING_MODEL", "text-embedding-3-small")

    # ----- LLM backend: "openai" or "local" -----
    LLM_BACKEND: str = os.getenv("ACT_IE_LLM_BACKEND", "openai")

    # ----- OpenAI backend settings -----
    OPENAI_MODEL: str = os.getenv("ACT_IE_OPENAI_MODEL", "gpt-4.1-mini")
    OPENAI_API_KEY_ENV: str = "OPENAI_API_KEY"

    # ----- Local HF model settings -----
    LOCAL_MODEL_NAME: str = os.getenv("ACT_IE_LOCAL_MODEL_NAME", "gpt2")

    # ----- Retrieval settings -----
    TOP_K_DOCS: int = int(os.getenv("ACT_IE_TOP_K_DOCS", "5"))

    # ----- Domain constants -----
    COUNTY_LIST: List[str] = [
        "Cork", "Kerry", "Limerick", "Tipperary", "Clare",
        "Galway", "Mayo", "Dublin", "Wexford", "Kilkenny",
        # ... extend for all 26 counties or NUTS3 regions
    ]

    SECTOR_LIST: List[str] = [
        "Dairy", "Beef", "Tillage", "Sheep", "Horticulture",
        "Forestry", "Mixed", "Logistics", "Processing", "Retail"
    ]

    # ----- Compliance / logging -----
    ENABLE_AUDIT_LOG: bool = os.getenv("ACT_IE_ENABLE_AUDIT_LOG", "true").lower() == "true"
    AUDIT_LOG_PATH: str = os.getenv("ACT_IE_AUDIT_LOG_PATH", "./act_ie_audit_log.jsonl")

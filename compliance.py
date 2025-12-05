# act_ie/compliance.py

import json
import time
from typing import Any, Dict, List

from .config import Config
from .models import RiskAssessment, RiskLevel


def classify_risk_level(query: str) -> RiskAssessment:
    """
    Very crude heuristic classifier for EU AI Act-ish risk tagging.
    You can refine based on keywords or patterns.
    """
    q_lower = query.lower()
    if "loan" in q_lower or "subsidy" in q_lower or "grant" in q_lower:
        level: RiskLevel = "high"
        reason = "Query appears related to financial or policy decisions affecting livelihoods."
    elif "investment" in q_lower or "compliance" in q_lower:
        level = "limited"
        reason = "Query touches on advisory/compliance but not direct high-risk allocation."
    else:
        level = "minimal"
        reason = "Analytical/insight query with no obvious high-risk decision binding."
    return RiskAssessment(level=level, reason=reason)


def audit_log(entry: Dict[str, Any]) -> None:
    if not Config.ENABLE_AUDIT_LOG:
        return
    try:
        entry_with_ts = {
            "timestamp": time.time(),
            **entry,
        }
        with open(Config.AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_with_ts) + "\n")
    except Exception as e:
        # Fail silently for now (you can later send this to proper logging)
        print(f"[ACT-IE][AUDIT] Failed to write log: {e}")


def build_audit_entry(
    query: str,
    county: str,
    sector: str,
    numeric_summary: str,
    retrieved_doc_ids: List[str],
    risk_assessment: RiskAssessment,
    answer: str,
) -> Dict[str, Any]:
    return {
        "query": query,
        "county": county,
        "sector": sector,
        "numeric_summary": numeric_summary,
        "retrieved_doc_ids": retrieved_doc_ids,
        "risk": {
            "level": risk_assessment.level,
            "reason": risk_assessment.reason,
        },
        "answer_preview": answer[:500],
    }

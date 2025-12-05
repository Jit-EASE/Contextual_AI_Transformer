# act_ie/geo_policy.py

from typing import Dict, Any
from .config import Config


# Very small stub – later you can load this from CSV / DB
_FAKE_POLICY_DB: Dict[str, Dict[str, Any]] = {
    # Example structure; you’ll replace with real data
    "Cork:Dairy": {
        "nitrates_band": "high",
        "acres_zone": "general",
        "notes": "High stocking density; derogation matters.",
    },
    "Kerry:Dairy": {
        "nitrates_band": "medium",
        "acres_zone": "cooperation",
        "notes": "Environmentally sensitive grassland in many regions.",
    },
}


def lookup_geo_policy(county: str, sector: str) -> Dict[str, Any]:
    """
    Minimal geo-policy adapter. Returns a dict you can inject into prompts.
    """
    key = f"{county}:{sector}"
    base = {
        "nitrates_band": "unknown",
        "acres_zone": "unknown",
        "notes": "No specific geo-policy metadata in this stub.",
    }
    return _FAKE_POLICY_DB.get(key, base)

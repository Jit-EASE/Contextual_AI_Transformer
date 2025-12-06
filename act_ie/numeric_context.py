# act_ie/numeric_context.py

from .models import NumericContext


def summarise_numeric_features(
    numeric: NumericContext,
    county: str,
    sector: str
) -> str:
    """
    Convert numeric features into a compact textual summary for the LLM.
    This is the v0 way to bind numbers into the prompt.
    """
    lines = [
        f"Region: {county} | Sector: {sector}",
        f"Milk price (EUR/litre): {numeric.milk_price_eur_per_litre:.3f}",
        f"12-month milk price volatility (std dev, EUR/litre): {numeric.milk_price_volatility_12m:.4f}",
        f"Rainfall anomaly (last 6 months, mm vs normal): {numeric.rainfall_anomaly_last_6m_mm:.1f}",
        f"NDVI mean (last season, 0–1): {numeric.ndvi_mean_last_season:.3f}",
        f"Representative herd size (number of cows): {numeric.herd_size:d}",
    ]
    return "\n".join(lines)


# Placeholder for future time-series / anomaly analysis
def classify_rainfall_anomaly(anomaly_mm: float) -> str:
    """
    Simple qualitative classification to inject into prompts later.
    """
    if anomaly_mm <= -80:
        return "severe deficit"
    elif anomaly_mm <= -40:
        return "moderate deficit"
    elif anomaly_mm >= 80:
        return "severe surplus"
    elif anomaly_mm >= 40:
        return "moderate surplus"
    else:
        return "near-normal"

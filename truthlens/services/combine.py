"""
Combines the three independent signals (ML model, Gemini reasoning,
NewsAPI corroboration) into a single final credibility verdict.
Any signal that is unavailable (e.g. missing API key) is simply
excluded and the remaining weights are renormalized.
"""
from config import Config


def combine_signals(ml_result: dict, gemini_result: dict, news_result: dict) -> dict:
    scores = []
    weights = []

    if ml_result.get("available"):
        scores.append(ml_result["credibility_score"])
        weights.append(Config.WEIGHT_ML_MODEL)

    if gemini_result.get("available"):
        scores.append(gemini_result["credibility_score"])
        weights.append(Config.WEIGHT_GEMINI)

    if news_result.get("available") and news_result.get("corroboration_score") is not None:
        scores.append(news_result["corroboration_score"])
        weights.append(Config.WEIGHT_NEWS_CORROBORATION)

    if not scores:
        return {
            "final_score": None,
            "final_label": "UNKNOWN",
            "note": "No analysis sources were available. Check your API keys and trained model.",
        }

    total_weight = sum(weights)
    normalized = [w / total_weight for w in weights]
    final_score = round(sum(s * w for s, w in zip(scores, normalized)), 1)

    if final_score >= 65:
        label = "CREDIBLE"
    elif final_score >= 40:
        label = "UNCERTAIN"
    else:
        label = "SUSPICIOUS"

    return {
        "final_score": final_score,
        "final_label": label,
        "sources_used": len(scores),
    }

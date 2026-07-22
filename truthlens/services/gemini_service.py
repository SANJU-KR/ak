"""
Uses Google's Gemini API to produce a human-readable credibility assessment
with reasoning — complementing the statistical ML model with an LLM's
contextual/world-knowledge judgment.
"""
import json
import re
import google.generativeai as genai
from config import Config

_configured = False


def _ensure_configured():
    global _configured
    if not _configured and Config.GEMINI_API_KEY:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        _configured = True


PROMPT_TEMPLATE = """You are a fact-checking assistant helping assess the credibility of a news headline or claim.

Claim to assess:
"{claim}"

Additional context you can use:
- Statistical ML model verdict: {ml_label} (confidence {ml_confidence}%)
- Live news search found {news_count} related articles ({reputable_count} from well-known outlets like Reuters/AP/BBC).

Respond with ONLY a valid JSON object (no markdown fences, no extra text) with exactly these fields:
{{
  "verdict": one of "Likely Credible", "Uncertain / Needs Verification", "Likely Misinformation",
  "credibility_score": integer 0-100 (100 = fully credible),
  "explanation": a 2-3 sentence plain-English explanation of your reasoning,
  "key_reasons": array of 2-4 short strings, the main factors behind your verdict,
  "red_flags": array of 0-4 short strings naming any manipulative language, unverifiable claims, or conspiratorial framing found (empty array if none)
}}
"""


def _extract_json(raw: str) -> dict:
    # Strip markdown code fences if the model adds them anyway
    cleaned = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def get_ai_analysis(claim: str, ml_label: str, ml_confidence: float,
                     news_count: int, reputable_count: int) -> dict:
    if not Config.GEMINI_API_KEY:
        return {
            "available": False,
            "error": "GEMINI_API_KEY not configured on the server (see .env.example).",
        }

    _ensure_configured()

    prompt = PROMPT_TEMPLATE.format(
        claim=claim,
        ml_label=ml_label,
        ml_confidence=round(ml_confidence * 100, 1) if ml_confidence <= 1 else ml_confidence,
        news_count=news_count,
        reputable_count=reputable_count,
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        parsed = _extract_json(response.text)
        parsed["available"] = True
        return parsed
    except json.JSONDecodeError:
        return {
            "available": False,
            "error": "Gemini returned a response that couldn't be parsed as JSON.",
        }
    except Exception as e:
        return {
            "available": False,
            "error": f"Gemini API request failed: {e}",
        }

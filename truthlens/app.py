"""
Backend — Flask API that powers real-time news credibility analysis.

Pipeline for POST /api/analyze:
  1. Run the trained TF-IDF + Logistic Regression model on the text.
  2. Search Newsdata.io for live related coverage and score corroboration.
  3. Ask Gemini for a contextual credibility assessment + explanation.
  4. Combine all three signals into one final verdict.
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from config import Config
from services.ml_predictor import ml_predictor
from services.news_service import get_related_articles
from services.gemini_service import get_ai_analysis
from services.combine import combine_signals

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "ml_model_loaded": ml_predictor.loaded,
        "news_api_configured": bool(Config.NEWS_API_KEY),
        "gemini_configured": bool(Config.GEMINI_API_KEY),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if len(text) < 4:
        return jsonify({"success": False, "error": "Please provide a longer headline or statement."}), 400

    # 1. ML model
    ml_result = ml_predictor.predict(text)

    # 2. Newsdata.io corroboration
    news_result = get_related_articles(text)

    # 3. Gemini reasoning (feed it the ML + news context for a better-informed answer)
    gemini_result = get_ai_analysis(
        claim=text,
        ml_label=ml_result.get("label", "UNKNOWN"),
        ml_confidence=ml_result.get("confidence", 0),
        news_count=news_result.get("total_results", 0),
        reputable_count=news_result.get("reputable_hits", 0),
    )

    # 4. Combine
    final_verdict = combine_signals(ml_result, gemini_result, news_result)

    return jsonify({
        "success": True,
        "input_text": text,
        "ml_prediction": ml_result,
        "news_corroboration": news_result,
        "ai_analysis": gemini_result,
        "final_verdict": final_verdict,
    })


if __name__ == "__main__":
    app.run(debug=Config.DEBUG, port=Config.PORT, host="0.0.0.0")

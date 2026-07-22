"""
Central configuration for the backend.
Loads secrets/settings from a local .env file (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))

    MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

    # Weighting used to combine the three signals into one final score
    WEIGHT_ML_MODEL = 0.40
    WEIGHT_GEMINI = 0.40
    WEIGHT_NEWS_CORROBORATION = 0.20

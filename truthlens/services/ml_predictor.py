"""
Wraps the trained scikit-learn TF-IDF + Logistic Regression model.
"""
import os
import joblib
from config import Config


class MLPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.loaded = False
        self._load()

    def _load(self):
        if os.path.exists(Config.MODEL_PATH) and os.path.exists(Config.VECTORIZER_PATH):
            self.model = joblib.load(Config.MODEL_PATH)
            self.vectorizer = joblib.load(Config.VECTORIZER_PATH)
            self.loaded = True
        else:
            self.loaded = False

    def predict(self, text: str) -> dict:
        """
        Returns a dict with label, confidence and both class probabilities.
        Label mapping: 0 = FAKE, 1 = REAL (see model/train_model.py)
        """
        if not self.loaded:
            return {
                "available": False,
                "error": (
                    "No trained model found. Run 'python model/train_model.py' "
                    "first (see README.md)."
                ),
            }

        vec = self.vectorizer.transform([text])
        proba = self.model.predict_proba(vec)[0]  # [P(fake), P(real)]
        fake_p, real_p = float(proba[0]), float(proba[1])
        label = "REAL" if real_p >= fake_p else "FAKE"

        return {
            "available": True,
            "label": label,
            "real_probability": round(real_p, 4),
            "fake_probability": round(fake_p, 4),
            "confidence": round(max(real_p, fake_p), 4),
            "credibility_score": round(real_p * 100, 1),
        }


# Singleton instance reused across requests
ml_predictor = MLPredictor()

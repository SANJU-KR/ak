# Real-Time News Credibility Analysis

This is your original front end (unchanged design/UI), now backed by a real
Flask API that combines three independent signals for every claim you submit:

1. **A trained ML model** (TF‑IDF + Logistic Regression, scikit-learn) — statistical
   fake/real classification.
2. **Newsdata.io** — searches live news coverage to see whether credible outlets are
   actually reporting on the claim (corroboration signal).
3. **Gemini API** — an LLM reasoning pass that explains *why* the claim looks
   credible or not, and flags manipulative language.

The three scores are combined into one final credibility verdict shown in the UI.
The original in-browser demo model is still used only to power the "Top Weighted
Tokens" panel — that part was always cosmetic and needs no server.

---

## 1. Project structure

```
truthlens/
├── app.py                     # Flask app + /api/analyze endpoint
├── config.py                  # Loads settings/keys from .env
├── requirements.txt
├── .env.example                # Copy to .env and fill in your keys
├── model/
│   ├── train_model.py          # Trains & saves the ML model
│   ├── sample_dataset.csv      # Small starter dataset (works out of the box)
│   ├── fake_news_model.pkl     # Created after you run train_model.py
│   └── tfidf_vectorizer.pkl    # Created after you run train_model.py
├── services/
│   ├── ml_predictor.py         # Loads model, runs predictions
│   ├── news_service.py         # Newsdata.io integration
│   ├── gemini_service.py       # Gemini API integration
│   └── combine.py              # Combines the 3 signals into 1 verdict
├── templates/
│   └── index.html              # Your original UI, wired to call /api/analyze
└── static/                     # (reserved for any future static assets)
```

---

## 2. Prerequisites

- Python 3.9+
- A free **Newsdata.io** key: https://newsdata.io/register (already configured
  in `.env` for you — see step 4 below)
- A free **Gemini** API key: https://aistudio.google.com/app/apikey

Both services work without hard failure if a key is missing — the corresponding
panel in the UI will just say "Offline" and explain why, and the final score is
recalculated using only the signals that are available.

---

## 3. Setup — step by step

```bash
# 1. Move into the project folder
cd truthlens

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API keys
cp .env.example .env
# open .env in an editor and paste in your real NEWS_API_KEY and GEMINI_API_KEY

# 5. Train the ML model (creates model/fake_news_model.pkl)
python model/train_model.py
```

That last step trains on the small bundled `sample_dataset.csv` (~50 labeled
headlines) so everything works immediately — accuracy will be modest since the
dataset is tiny. For a much stronger model:

```bash
# Optional: train on the full ISOT Fake/Real News dataset (~44,000 articles)
# 1. Download Fake.csv and True.csv from:
#    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# 2. Place both files inside the model/ folder
python model/train_model.py --full
```

---

## 4. Run the server

```bash
python app.py
```

Then open **http://localhost:5000** in your browser. The original UI
loads exactly as before — type or pick a sample headline and click **Analyze →**.
It will now call the real backend and show:

- The combined **Real-Time Verdict** and credibility score at the top
- An **ML Model** panel with the trained classifier's probabilities
- An **AI Reasoning — Gemini** panel with an explanation and flagged red flags
- A **Live News Coverage** panel with real, clickable related articles
- The original **Top Weighted Tokens** panel (local demo model, unchanged)

---

## 5. API reference

### `POST /api/analyze`
Request body:
```json
{ "text": "Scientists develop new malaria vaccine with 90% efficacy in trials" }
```

Response body (shape):
```json
{
  "success": true,
  "input_text": "...",
  "ml_prediction": { "available": true, "label": "REAL", "real_probability": 0.87, "fake_probability": 0.13, "credibility_score": 87.0 },
  "news_corroboration": { "available": true, "total_results": 5, "reputable_hits": 3, "corroboration_score": 71.0, "articles": [ ... ] },
  "ai_analysis": { "available": true, "verdict": "Likely Credible", "credibility_score": 82, "explanation": "...", "key_reasons": [...], "red_flags": [] },
  "final_verdict": { "final_score": 83.4, "final_label": "CREDIBLE", "sources_used": 3 }
}
```

### `GET /api/health`
Quick check of whether the model is loaded and which API keys are configured.

---

## 6. Troubleshooting

- **"No trained model found"** → run `python model/train_model.py` first.
- **News panel says "Offline"** → check `NEWS_API_KEY` in `.env`; the free tier
  also has a request-per-day limit.
- **Gemini panel says "Offline"** → check `GEMINI_API_KEY` in `.env`; make sure
  the `google-generativeai` package installed correctly.
- **CORS errors when calling from a different origin** → `flask-cors` is already
  enabled for all origins in `app.py`; adjust `CORS(app)` if you need to restrict it.

---

## 7. Deploying

The repo already includes what each platform needs — pick one.

### Option A — Render (easiest, free tier, no Docker needed)
1. Push this folder to a GitHub repo.
2. Go to https://dashboard.render.com → **New** → **Blueprint**, and point it at
   your repo. Render will read `render.yaml` automatically and configure the
   service (build command trains the model, start command runs gunicorn).
3. In the service's **Environment** tab, add your real values for
   `NEWS_API_KEY` and `GEMINI_API_KEY` (left blank in `render.yaml` on purpose —
   never commit real keys to git).
4. Click **Deploy**. Render gives you a public `https://<name>.onrender.com` URL.

   No Blueprint UI? Create a Web Service manually instead:
   - Build command: `pip install -r requirements.txt && python model/train_model.py`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - Add the same two environment variables.

### Option B — Railway / Fly.io / Google Cloud Run (Docker)
The included `Dockerfile` trains the sample model at build time and serves with
gunicorn, so any Docker-based host works the same way:

```bash
# Railway
railway init
railway up
railway variables set NEWS_API_KEY=... GEMINI_API_KEY=...

# Fly.io
fly launch          # detects the Dockerfile
fly secrets set NEWS_API_KEY=... GEMINI_API_KEY=...
fly deploy

# Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/news-credibility
gcloud run deploy --image gcr.io/PROJECT_ID/news-credibility \
  --set-env-vars NEWS_API_KEY=...,GEMINI_API_KEY=...
```

Or run the container anywhere yourself:
```bash
docker build -t news-credibility .
docker run -p 5000:5000 -e NEWS_API_KEY=... -e GEMINI_API_KEY=... news-credibility
```

### Option C — Heroku
The `Procfile` and `runtime.txt` are already set up:
```bash
heroku create your-app-name
heroku config:set NEWS_API_KEY=... GEMINI_API_KEY=...
git push heroku main
heroku run python model/train_model.py   # trains the model on the dyno once
```

### A note on the ML model file in production
`fake_news_model.pkl` / `tfidf_vectorizer.pkl` are trained during the build step
(they're in `.gitignore` on purpose — don't commit binary model files to git).
If your platform's filesystem resets between deploys but not within a running
instance, that's fine — the build step re-trains automatically every deploy.
For the stronger `--full` model in production, commit `Fake.csv`/`True.csv` into
`model/` (or fetch them in the build step) and change the build/Docker command
to `python model/train_model.py --full`.


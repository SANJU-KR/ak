"""
Cross-references the submitted claim against live news coverage using
Newsdata.io (https://newsdata.io). This helps answer: "is any credible
outlet actually reporting this?" - a real corroboration signal fake/invented
stories lack.
"""
import re
import requests
from config import Config

NEWSDATA_URL = "https://newsdata.io/api/1/news"

# A short list of generally high-credibility outlets used only to weight
# corroboration - NOT used to censor or exclude any other source. Matched
# against both the article domain and Newsdata's source_id slug.
REPUTABLE_SOURCES = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "nytimes.com", "washingtonpost.com", "theguardian.com", "wsj.com",
    "bloomberg.com", "cnn.com", "abcnews.go.com", "cbsnews.com",
    "nbcnews.com", "aljazeera.com", "economist.com", "ft.com",
    "reuters", "apnews", "bbc", "npr", "nytimes", "washingtonpost",
    "theguardian", "wsj", "bloomberg", "cnn", "aljazeera",
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "for", "and", "or", "with", "that", "this", "it", "as", "by",
    "from", "has", "have", "had", "be", "been", "will", "after", "over",
}


def _extract_query(text: str, max_words: int = 6) -> str:
    """Pull the most meaningful keywords out of the claim for a search query."""
    words = re.findall(r"[A-Za-z0-9']+", text)
    keywords = [w for w in words if len(w) > 3 and w.lower() not in STOPWORDS]
    seen = []
    for w in keywords:
        if w.lower() not in [s.lower() for s in seen]:
            seen.append(w)
        if len(seen) >= max_words:
            break
    return " ".join(seen) if seen else text[:80]


def get_related_articles(text: str, page_size: int = 6) -> dict:
    if not Config.NEWS_API_KEY:
        return {
            "available": False,
            "error": "NEWS_API_KEY not configured on the server (see .env.example).",
            "articles": [],
            "corroboration_score": None,
        }

    query = _extract_query(text)

    try:
        resp = requests.get(
            NEWSDATA_URL,
            params={
                "apikey": Config.NEWS_API_KEY,
                "q": query,
                "language": "en",
            },
            timeout=10,
        )
        data = resp.json()
    except requests.RequestException as e:
        return {
            "available": False,
            "error": f"Newsdata.io request failed: {e}",
            "articles": [],
            "corroboration_score": None,
        }

    if data.get("status") != "success":
        return {
            "available": False,
            "error": (data.get("results") or {}).get("message", None)
                      or data.get("message", "Unknown Newsdata.io error"),
            "articles": [],
            "corroboration_score": None,
        }

    raw_articles = (data.get("results") or [])[:page_size]

    articles = []
    reputable_hits = 0
    for a in raw_articles:
        url = a.get("link", "") or ""
        domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        source_id = (a.get("source_id") or "").lower()
        is_reputable = domain in REPUTABLE_SOURCES or source_id in REPUTABLE_SOURCES
        if is_reputable:
            reputable_hits += 1
        articles.append({
            "title": a.get("title"),
            "source": a.get("source_id") or domain or "Unknown",
            "url": url,
            "publishedAt": a.get("pubDate"),
            "description": a.get("description"),
            "reputable_source": is_reputable,
        })

    total = len(articles)
    if total == 0:
        corroboration_score = 20  # no coverage found at all is a mild red flag
    else:
        coverage_score = min(total / page_size, 1.0) * 60
        reputable_ratio = (reputable_hits / total) if total else 0
        corroboration_score = round(coverage_score + reputable_ratio * 40, 1)

    return {
        "available": True,
        "query_used": query,
        "total_results": total,
        "reputable_hits": reputable_hits,
        "corroboration_score": corroboration_score,
        "articles": articles,
    }

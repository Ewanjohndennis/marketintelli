import os
import re
from collections import Counter
from dotenv import load_dotenv
from serpapi import GoogleSearch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spacy

from mcp.server.fastmcp import FastMCP

load_dotenv()

SERP_API_KEY = os.getenv("SERP_API_KEY")
if not SERP_API_KEY:
    raise EnvironmentError(
        "SERP_API_KEY not set. Add it to your .env file or environment variables."
    )

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

mcp = FastMCP("Market Research Tools")

# Noise words that spaCy sometimes includes at the start/end of an entity span
_STRIP_WORDS = {
    "the", "a", "an", "shop", "buy", "get", "find", "best",
    "top", "new", "men", "women", "kids", "sale", "online",
}

def _normalize(name: str) -> str:
    """
    Clean a raw spaCy entity down to its core brand name.
      - Lowercase-strip leading/trailing noise words
      - Remove possessives and punctuation suffixes
      - Strip URLs to domain root (Nike.com → Nike)
      - Title-case the result
    """
    # Remove possessive
    name = re.sub(r"'[sS]?\s*$", "", name).strip()

    # Strip URL suffix so Nike.com → Nike
    name = re.sub(r"\.com.*$", "", name, flags=re.IGNORECASE).strip()

    # Remove leading/trailing noise words (case-insensitive, word by word)
    words = name.split()
    while words and words[0].lower() in _STRIP_WORDS:
        words.pop(0)
    while words and words[-1].lower() in _STRIP_WORDS:
        words.pop()

    return " ".join(words).title() if words else ""


@mcp.tool()
def market_graph(counts: dict) -> str:
    """
    Generate and save a bar chart of brand frequency.
    Returns the file path of the saved image.
    """
    brands = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(brands, values, color="steelblue", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val),
            ha="center", va="bottom", fontsize=11
        )

    ax.set_title("Brand Mentions in Market Search Results", fontsize=14, pad=12)
    ax.set_xlabel("Brand")
    ax.set_ylabel("Mentions")
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    path = "market_graph.png"
    plt.savefig(path, dpi=150)
    plt.close()

    return f"Graph saved to {path}"


@mcp.tool()
def brand_frequency(data: list) -> dict:
    """
    Extract brand names from search results using spaCy NER.
    Normalizes entities to remove noise words and merge duplicates.
    Returns the top 10 most frequent brands.
    """
    brands = []

    for item in data:
        text = item.get("title", "") + " " + item.get("snippet", "")
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT"):
                name = _normalize(ent.text)
                if name and len(name) > 1:
                    brands.append(name)

    counts = Counter(brands)
    top_brands = dict(counts.most_common(10))

    return top_brands


@mcp.tool()
def search_market(keyword: str) -> list:
    """
    Search Google for market information about a product or company.
    Returns structured search results (title, snippet, link).
    """
    params = {
        "q": keyword,
        "engine": "google",
        "hl": "en",
        "gl": "us",
        "num": 10,
        "api_key": SERP_API_KEY,
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic = results.get("organic_results", [])

    data = []
    for r in organic[:10]:
        entry = {
            "title":   r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "link":    r.get("link", ""),
        }
        data.append(entry)

    return data


if __name__ == "__main__":
    mcp.run()
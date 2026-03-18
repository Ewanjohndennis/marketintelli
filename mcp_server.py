import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf = os.getenv("HF_TOKEN")
news_key = os.getenv("NEWS_API_KEY")

# LLM
endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    max_new_tokens=800,
    temperature=0.5
)

llm = ChatHuggingFace(llm=endpoint)

newsapi = NewsApiClient(api_key=news_key)

# MCP server
mcp = FastMCP("Market Intelligence Tools")

# ---------------------------
# NEWS TOOL
# ---------------------------
@mcp.tool()
def get_news(topic: str) -> str:
    """Fetch latest news articles about a company or industry."""

    articles = newsapi.get_everything(
        q=topic,
        language="en",
        sort_by="publishedAt",
        page_size=5
    )

    text = ""

    for a in articles["articles"]:
        title = a["title"] or ""
        desc = a["description"] or ""
        date = a["publishedAt"]

        date = datetime.fromisoformat(date.replace("Z","")).strftime("%Y-%m-%d")

        text += f"[{date}] {title}. {desc}\n\n"

    return text


# ---------------------------
# MARKET ANALYSIS TOOL
# ---------------------------
@mcp.tool()
def analyze_market(news_text: str) -> str:
    """Analyze news and produce a market insight paragraph."""

    prompt = f"""
You are a market intelligence analyst.

Analyze the following news articles and write ONE clear paragraph summarizing the market insight.

Mention:
- sector involved
- company discussed
- competitors
- dominant companies
- opportunity or threat

News:
{news_text}
"""

    response = llm.invoke(prompt)

    return response.content


# Run MCP server
if __name__ == "__main__":
    mcp.run()
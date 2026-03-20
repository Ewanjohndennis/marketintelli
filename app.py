import io, json, os, time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from serpapi import GoogleSearch
from openai import OpenAI
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak,
)
load_dotenv()

# ── yfinance crumb fix — must be set before any yfinance calls ─────────────────
import requests as _requests
_YF_SESSION = _requests.Session()
_YF_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

# ── Config ─────────────────────────────────────────────────────────────────────
SERP_API_KEY       = st.secrets["SERP_API_KEY"]
MONGO_URI          = st.secrets["MONGO_URI"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

groq_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://marketintelli.streamlit.app",
        "X-Title": "Market Intelligence Dashboard",
    }
)

COLORS = [
    ("#4F8EF7", "rgba(79,142,247,0.12)"),
    ("#F76A4F", "rgba(247,106,79,0.12)"),
    ("#2ECC71", "rgba(46,204,113,0.12)"),
    ("#F1C40F", "rgba(241,196,15,0.12)"),
    ("#9B59B6", "rgba(155,89,182,0.12)"),
]

# ══════════════════════════════════════════════════════════════════════════════
# MongoDB connection
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_db():
    """Cached MongoDB connection — reused across all user sessions."""
    client = MongoClient(MONGO_URI)
    return client["market_intelligence"]

# ══════════════════════════════════════════════════════════════════════════════
# 1. AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════
USERS = {
    "admin":    {"password": "admin", "role": "admin"},
    "employee": {"password": "1234",  "role": "employee"},
}

def show_login():
    """Show login screen — blocks until authenticated."""
    st.title("🧠 Company Intelligence Dashboard")
    st.divider()
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.subheader("Sign In")
        user_id  = st.text_input("User ID", placeholder="employee or admin")
        password = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            user = USERS.get(user_id.strip())
            if user and user["password"] == password.strip():
                st.session_state["logged_in"] = True
                st.session_state["user_id"]   = user_id.strip()
                st.session_state["role"]      = user["role"]
                st.rerun()
            else:
                st.error("Invalid user ID or password.")
    return False

def is_admin() -> bool:
    return st.session_state.get("role") == "admin"

# ══════════════════════════════════════════════════════════════════════════════
# Shared settings — stored in MongoDB, visible to ALL users instantly
# ══════════════════════════════════════════════════════════════════════════════
CONFIG_DOC_ID = "global_company_config"

def load_settings() -> dict:
    try:
        db  = get_db()
        doc = db["config"].find_one({"_id": CONFIG_DOC_ID})
        if doc:
            return {
                "company_name":     doc.get("company_name", ""),
                "company_ticker":   doc.get("company_ticker", ""),
                "competitors":      doc.get("competitors", []),
                "auto_competitors": doc.get("auto_competitors", True),
            }
    except Exception:
        pass
    return {
        "company_name":     "",
        "company_ticker":   "",
        "competitors":      [],
        "auto_competitors": True,
    }

def save_settings(s: dict):
    try:
        db = get_db()
        db["config"].update_one(
            {"_id": CONFIG_DOC_ID},
            {"$set": {
                "company_name":     s["company_name"],
                "company_ticker":   s["company_ticker"],
                "competitors":      s["competitors"],
                "auto_competitors": s["auto_competitors"],
                "updated_at":       datetime.utcnow(),
                "updated_by":       st.session_state.get("user_id", "admin"),
            }},
            upsert=True,
        )
    except Exception as e:
        st.error(f"Failed to save settings to MongoDB: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. AUTO TICKER RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════
KNOWN_TICKERS = {
    "apple": "AAPL", "samsung": "005930.KS", "google": "GOOGL",
    "alphabet": "GOOGL", "microsoft": "MSFT", "amazon": "AMZN",
    "meta": "META", "facebook": "META", "tesla": "TSLA",
    "nvidia": "NVDA", "nike": "NKE", "adidas": "ADDYY",
    "puma": "PUMSY", "sony": "SONY", "intel": "INTC",
    "amd": "AMD", "netflix": "NFLX", "spotify": "SPOT",
    "uber": "UBER", "airbnb": "ABNB", "coca cola": "KO",
    "pepsi": "PEP", "walmart": "WMT", "disney": "DIS",
    "ford": "F", "toyota": "TM", "jpmorgan": "JPM",
    "tata": "TATAMOTORS.NS", "infosys": "INFY", "wipro": "WIT",
    "reliance": "RELIANCE.NS", "hdfc": "HDFCBANK.NS",
}

def resolve_ticker(keyword: str) -> str | None:
    key = keyword.lower().strip()
    if key in KNOWN_TICKERS:
        return KNOWN_TICKERS[key]
    try:
        results = yf.Search(keyword, max_results=1).quotes
        if results:
            return results[0].get("symbol")
    except Exception:
        pass
    return None

def auto_detect_competitors(company_name: str) -> list:
    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="nvidia/nemotron-3-nano-30b-a3b:free",
                messages=[
                    {"role": "system", "content":
                        "Return ONLY a JSON array of exactly 4 top competitor company names. "
                        "No explanation, no markdown, just the JSON array."},
                    {"role": "user", "content": f"Top 4 competitors of {company_name}"}
                ],
                max_tokens=150, temperature=0.3,
            )
            raw     = response.choices[0].message.content.strip()
            cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)
        except Exception:
            time.sleep(2 ** attempt)
    return []

# ══════════════════════════════════════════════════════════════════════════════
# Agent prompts
# ══════════════════════════════════════════════════════════════════════════════
AGENTS = {
    "news_analyst":
        "You are NewsAnalyst for an internal company intelligence system. "
        "Analyse the latest news headlines. Summarise key developments, risks, "
        "and opportunities in 2 short paragraphs. Be direct and actionable. "
        "Be concise. Do not repeat the question. Start your response directly.",
    "competitor_analyst":
        "You are CompetitorAnalyst. Compare the company against its competitors "
        "based on search trends and news. Who is gaining ground? Who is losing? "
        "What are the key differentiators? 2 short paragraphs. "
        "Be concise. Do not repeat the question. Start your response directly.",
    "financial_analyst":
        "You are FinancialAnalyst. Analyse the stock performance and revenue data. "
        "Identify growth trends, risks, and financial health signals. "
        "Be concise and data-driven. 2 short paragraphs. "
        "Be concise. Do not repeat the question. Start your response directly.",
    "improvement_analyst":
        "You are StrategicAdvisor. Based on the news, competitor positioning, "
        "trends, and financials, identify 5 specific actionable improvements. "
        "Format as a numbered list. Be direct and specific. "
        "Be concise. Do not repeat the question. Start your response directly.",
    "synthesizer":
        "You are the Chief Intelligence Officer. Merge all analyst inputs into "
        "an executive brief with these sections:\n"
        "1. Company Pulse\n2. Competitive Position\n3. Financial Health\n"
        "4. Key Risks\n5. Strategic Recommendations\n"
        "Be sharp, concise, and executive-ready. Use bullet points. "
        "Be concise. Do not repeat the question. Start your response directly.",
}

def call_agent(role: str, message: str, retries: int = 3) -> str:
    """Call OpenRouter with automatic retry on transient connection errors."""
    last_err = None
    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model="nvidia/nemotron-3-nano-30b-a3b:free",
                messages=[
                    {"role": "system", "content": AGENTS[role]},
                    {"role": "user",   "content": message},
                ],
                max_tokens=1500, temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return f"[Analysis unavailable — OpenRouter error after {retries} retries: {str(last_err)[:120]}]"

# ══════════════════════════════════════════════════════════════════════════════
# Data fetchers
# ══════════════════════════════════════════════════════════════════════════════
def fetch_trend(keyword: str) -> pd.DataFrame:
    try:
        data = GoogleSearch({
            "engine": "google_trends", "q": keyword,
            "data_type": "TIMESERIES", "date": "today 12-m",
            "api_key": SERP_API_KEY,
        }).get_dict()
        rows = [
            {"date": p["date"], "value": p["values"][0].get("extracted_value", 0)}
            for p in data.get("interest_over_time", {}).get("timeline_data", [])
        ]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def fetch_news(keyword: str, num: int = 6) -> list:
    try:
        data = GoogleSearch({
            "engine": "google", "q": f"{keyword} news",
            "tbm": "nws", "num": num, "api_key": SERP_API_KEY,
        }).get_dict()
        return [
            {
                "title":   r.get("title", ""),
                "source":  r.get("source", ""),
                "date":    r.get("date", ""),
                "snippet": r.get("snippet", ""),
                "link":    r.get("link", ""),
            }
            for r in data.get("news_results", [])
        ]
    except Exception:
        return []

def fetch_stock_price(ticker: str) -> pd.DataFrame:
    """Fetch 12 months daily price via yfinance with session crumb fix."""
    try:
        t  = yf.Ticker(ticker, session=_YF_SESSION)
        df = t.history(period="1y")[["Close"]].reset_index()
        df.columns = ["date", "price"]
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df
    except Exception:
        return pd.DataFrame()

def fetch_financials(ticker: str) -> dict:
    """Fetch key financials via yfinance with session crumb fix."""
    try:
        t     = yf.Ticker(ticker, session=_YF_SESSION)
        info  = t.info
        q_fin = t.quarterly_financials
        revenue_series = {}
        if q_fin is not None and not q_fin.empty:
            rev_row = q_fin.loc["Total Revenue"] if "Total Revenue" in q_fin.index else None
            if rev_row is not None:
                revenue_series = {
                    str(col.date()): int(val / 1e6)
                    for col, val in rev_row.items() if str(val) != "nan"
                }
        return {
            "ticker":            ticker.upper(),
            "company_name":      info.get("longName", ticker),
            "market_cap":        info.get("marketCap", 0),
            "revenue_ttm":       info.get("totalRevenue", 0),
            "gross_margin":      info.get("grossMargins", 0),
            "pe_ratio":          info.get("trailingPE", 0),
            "52w_high":          info.get("fiftyTwoWeekHigh", 0),
            "52w_low":           info.get("fiftyTwoWeekLow", 0),
            "current_price":     info.get("currentPrice", 0),
            "revenue_quarterly": revenue_series,
            "debt_to_equity":    info.get("debtToEquity", 0),
            "return_on_equity":  info.get("returnOnEquity", 0),
            "operating_margin":  info.get("operatingMargins", 0),
            "eps":               info.get("trailingEps", 0),
            "dividend_yield":    info.get("dividendYield", 0),
        }
    except Exception:
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# Chart builders
# ══════════════════════════════════════════════════════════════════════════════
def build_trend_chart(keywords, dataframes):
    fig = go.Figure()
    for i, (keyword, df) in enumerate(zip(keywords, dataframes)):
        if df.empty: continue
        color, fill = COLORS[i % len(COLORS)]
        vals = df["value"].astype(int)
        fig.add_trace(go.Scatter(
            x=df["date"], y=vals, name=keyword,
            mode="lines", line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=fill,
            hovertemplate="%{x} — <b>%{y}</b><extra>" + keyword + "</extra>",
        ))
        pk = vals.idxmax()
        fig.add_trace(go.Scatter(
            x=[df.loc[pk, "date"]], y=[vals[pk]], mode="markers+text",
            marker=dict(color=color, size=11, symbol="star"),
            text=[f" Peak {vals[pk]}"], textposition="middle right",
            textfont=dict(color=color), showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        hovermode="x unified", height=360,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(title="Interest (0-100)", gridcolor="rgba(180,180,180,0.15)"),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig

def build_stock_chart(tickers, stock_dfs):
    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        df = stock_dfs.get(ticker)
        if df is None or df.empty: continue
        color, _ = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["price"].round(2), name=ticker,
            mode="lines", line=dict(color=color, width=2.5),
            hovertemplate="%{x} — <b>$%{y}</b><extra>" + ticker + "</extra>",
        ))
    fig.update_layout(
        hovermode="x unified", height=340,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(title="Stock Price (USD)", gridcolor="rgba(180,180,180,0.15)"),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig

def build_revenue_chart(tickers, financials):
    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        fin = financials.get(ticker, {})
        rev = fin.get("revenue_quarterly", {})
        if not rev: continue
        dates  = sorted(rev.keys())
        values = [rev[d] for d in dates]
        color, _ = COLORS[i % len(COLORS)]
        fig.add_trace(go.Bar(
            x=dates, y=values,
            name=f"{fin.get('company_name', ticker)} ($M)",
            marker_color=color, opacity=0.85,
        ))
    fig.update_layout(
        barmode="group", height=300,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(title="Revenue ($M)", gridcolor="rgba(180,180,180,0.15)"),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 3. FINANCIAL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
def render_financial_comparison(ticker_map: dict, financials: dict):
    st.subheader("📊 Financial Metrics Comparison")
    metrics = [
        ("Current Price",    "current_price",   lambda v: f"${v:.2f}"),
        ("Market Cap",       "market_cap",       lambda v: f"${v/1e9:.1f}B"),
        ("TTM Revenue",      "revenue_ttm",      lambda v: f"${v/1e9:.1f}B"),
        ("Gross Margin",     "gross_margin",     lambda v: f"{v*100:.1f}%"),
        ("Operating Margin", "operating_margin", lambda v: f"{v*100:.1f}%"),
        ("P/E Ratio",        "pe_ratio",         lambda v: f"{v:.1f}x"),
        ("EPS",              "eps",              lambda v: f"${v:.2f}"),
        ("Return on Equity", "return_on_equity", lambda v: f"{v*100:.1f}%"),
        ("Debt / Equity",    "debt_to_equity",   lambda v: f"{v:.2f}"),
        ("Dividend Yield",   "dividend_yield",   lambda v: f"{v*100:.2f}%" if v else "N/A"),
        ("52W High",         "52w_high",         lambda v: f"${float(v):.2f}"),
        ("52W Low",          "52w_low",          lambda v: f"${float(v):.2f}"),
    ]
    rows = {}
    for label, key, fmt in metrics:
        row = {}
        for company, ticker in ticker_map.items():
            fin = financials.get(ticker, {})
            val = fin.get(key, 0)
            try:
                row[company] = fmt(val) if val else "N/A"
            except Exception:
                row[company] = "N/A"
        rows[label] = row
    df = pd.DataFrame(rows).T
    df.index.name = "Metric"
    st.dataframe(df, width="stretch")

    st.subheader("📈 Key Metrics — Visual Comparison")
    metric_choice = st.selectbox(
        "Select metric to compare",
        ["Market Cap ($B)", "TTM Revenue ($B)", "Gross Margin (%)",
         "P/E Ratio", "Return on Equity (%)", "Operating Margin (%)"],
    )
    metric_map = {
        "Market Cap ($B)":      ("market_cap",       lambda v: v / 1e9),
        "TTM Revenue ($B)":     ("revenue_ttm",      lambda v: v / 1e9),
        "Gross Margin (%)":     ("gross_margin",     lambda v: v * 100),
        "P/E Ratio":            ("pe_ratio",         lambda v: v),
        "Return on Equity (%)": ("return_on_equity", lambda v: v * 100),
        "Operating Margin (%)": ("operating_margin", lambda v: v * 100),
    }
    key, transform = metric_map[metric_choice]
    bar_fig = go.Figure()
    for i, (company, ticker) in enumerate(ticker_map.items()):
        fin = financials.get(ticker, {})
        val = fin.get(key, 0)
        color, _ = COLORS[i % len(COLORS)]
        bar_fig.add_trace(go.Bar(
            x=[company], y=[transform(val) if val else 0],
            name=company, marker_color=color,
        ))
    bar_fig.update_layout(
        height=320, barmode="group",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title=metric_choice, gridcolor="rgba(180,180,180,0.15)"),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(bar_fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf(company: str, competitors: list, brief: str,
                 news_out: str, comp_out: str, fin_out: str,
                 improve_out: str, financials: dict,
                 ticker_map: dict, generated_at: str) -> bytes:
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
    )
    styles = getSampleStyleSheet()
    story  = []
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"],
        fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
    h1_style = ParagraphStyle("H1", parent=styles["Heading1"],
        fontSize=14, spaceAfter=4, spaceBefore=12, textColor=colors.HexColor("#185FA5"))
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=10, spaceAfter=6, leading=15)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#666666"), spaceAfter=12)

    def add_section(title, content):
        story.append(Paragraph(title, h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#185FA5")))
        story.append(Spacer(1, 6))
        for line in content.split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 10))

    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"{company}", title_style))
    story.append(Paragraph("Intelligence Report", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {generated_at}", caption_style))
    if competitors:
        story.append(Paragraph(f"Benchmarked against: {', '.join(competitors)}", caption_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#185FA5")))
    story.append(Spacer(1, 0.2*inch))
    add_section("Executive Brief", brief)
    story.append(PageBreak())

    story.append(Paragraph("Financial Metrics", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#185FA5")))
    story.append(Spacer(1, 8))
    pdf_metrics = [
        ("Current Price",    "current_price",   lambda v: f"${v:.2f}"),
        ("Market Cap",       "market_cap",       lambda v: f"${v/1e9:.1f}B"),
        ("TTM Revenue",      "revenue_ttm",      lambda v: f"${v/1e9:.1f}B"),
        ("Gross Margin",     "gross_margin",     lambda v: f"{v*100:.1f}%"),
        ("Operating Margin", "operating_margin", lambda v: f"{v*100:.1f}%"),
        ("P/E Ratio",        "pe_ratio",         lambda v: f"{v:.1f}x"),
        ("EPS",              "eps",              lambda v: f"${v:.2f}"),
        ("Return on Equity", "return_on_equity", lambda v: f"{v*100:.1f}%"),
        ("52W High",         "52w_high",         lambda v: f"${float(v):.2f}"),
        ("52W Low",          "52w_low",          lambda v: f"${float(v):.2f}"),
    ]
    company_list = list(ticker_map.keys())
    table_data   = [["Metric"] + company_list]
    for label, key, fmt in pdf_metrics:
        row = [label]
        for comp in company_list:
            ticker = ticker_map.get(comp, "")
            fin    = financials.get(ticker, {})
            val    = fin.get(key, 0)
            try:
                row.append(fmt(val) if val else "N/A")
            except Exception:
                row.append("N/A")
        table_data.append(row)
    col_width = (6.5 * inch) / len(["Metric"] + company_list)
    tbl = Table(table_data, colWidths=[col_width] * len(["Metric"] + company_list))
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#185FA5")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  10),
        ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE",       (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f8ff"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("FONTNAME",       (0, 1), (0, -1),  "Helvetica-Bold"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 16))
    story.append(PageBreak())
    add_section("News & Sentiment Analysis", news_out)
    add_section("Competitor Analysis",       comp_out)
    add_section("Financial Analysis",        fin_out)
    story.append(PageBreak())
    add_section("Strategic Recommendations", improve_out)
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Paragraph(
        "Auto-generated by Company Intelligence Dashboard. "
        "Data sourced from Google Trends, Google News, and Yahoo Finance.",
        caption_style,
    ))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# ══════════════════════════════════════════════════════════════════════════════
# Main dashboard renderer
# ══════════════════════════════════════════════════════════════════════════════
def render_dashboard(settings: dict):
    company      = settings["company_name"]
    competitors  = settings["competitors"]
    all_keywords = [company] + competitors
    ticker_map   = {}

    with st.spinner("Resolving tickers automatically…"):
        t = resolve_ticker(company)
        if t:
            ticker_map[company] = t
        for comp in competitors:
            t = resolve_ticker(comp)
            if t:
                ticker_map[comp] = t

    if ticker_map:
        st.sidebar.success(
            "Tickers: " + ", ".join([f"{k} → `{v}`" for k, v in ticker_map.items()])
        )

    with st.spinner("Gathering intelligence…"):
        with ThreadPoolExecutor(max_workers=16) as ex:
            trend_futures     = [(kw, ex.submit(fetch_trend, kw)) for kw in all_keywords]
            news_future       = ex.submit(fetch_news, company, 6)
            comp_news_futures = [(c,  ex.submit(fetch_news, c, 3)) for c in competitors]
            stock_futures     = [(t,  ex.submit(fetch_stock_price, t)) for t in ticker_map.values()]
            fin_futures       = [(t,  ex.submit(fetch_financials, t)) for t in ticker_map.values()]

            trend_dfs    = {kw: f.result() for kw, f in trend_futures}
            company_news = news_future.result()
            comp_news    = {c: f.result() for c, f in comp_news_futures}
            stock_dfs    = {t: f.result() for t, f in stock_futures}
            financials   = {t: f.result() for t, f in fin_futures}

    company_fin    = financials.get(ticker_map.get(company, ""), {})
    news_headlines = " | ".join([a["title"] for a in company_news[:6]])
    comp_headlines = "\n".join([
        f"{c}: {' | '.join([a['title'] for a in articles[:3]])}"
        for c, articles in comp_news.items()
    ])
    trend_df      = trend_dfs.get(company, pd.DataFrame())
    trend_summary = (
        f"{company}: avg={trend_df['value'].mean():.1f}, peak={trend_df['value'].max()}"
        if not trend_df.empty else ""
    )
    comp_trends = "\n".join([
        f"{kw}: avg={df['value'].mean():.1f}, peak={df['value'].max()}"
        for kw, df in trend_dfs.items() if kw != company and not df.empty
    ])
    fin_summary = (
        f"Market Cap: ${company_fin.get('market_cap',0)/1e9:.1f}B, "
        f"TTM Revenue: ${company_fin.get('revenue_ttm',0)/1e9:.1f}B, "
        f"Gross Margin: {company_fin.get('gross_margin',0)*100:.1f}%, "
        f"P/E: {company_fin.get('pe_ratio',0):.1f}, "
        f"Price: ${company_fin.get('current_price',0):.2f}"
    ) if company_fin else ""

    full_context = (
        f"COMPANY: {company}\n\nNEWS:\n{news_headlines}\n\n"
        f"COMPETITOR NEWS:\n{comp_headlines}\n\n"
        f"TRENDS:\n{trend_summary}\n{comp_trends}\n\n"
        f"FINANCIALS:\n{fin_summary}"
    )

    with st.spinner("Running AI analysis…"):
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_news    = ex.submit(call_agent, "news_analyst",
                                  f"Company: {company}\n\nNews:\n{news_headlines}")
            f_comp    = ex.submit(call_agent, "competitor_analyst",
                                  f"Company: {company}\nCompetitors: {', '.join(competitors)}\n\n"
                                  f"Trends:\n{trend_summary}\n{comp_trends}\n\n"
                                  f"Competitor news:\n{comp_headlines}")
            f_fin     = ex.submit(call_agent, "financial_analyst",
                                  f"Company: {company}\nFinancials: {fin_summary}")
            f_improve = ex.submit(call_agent, "improvement_analyst", full_context)

            news_out    = f_news.result()
            comp_out    = f_comp.result()
            fin_out     = f_fin.result()
            improve_out = f_improve.result()

    with st.spinner("Synthesizing executive brief…"):
        brief = call_agent(
            "synthesizer",
            f"NewsAnalyst:\n{news_out}\n\nCompetitorAnalyst:\n{comp_out}\n\n"
            f"FinancialAnalyst:\n{fin_out}\n\nImprovementAnalyst:\n{improve_out}",
        )

    generated_at = datetime.now().strftime("%B %d, %Y %H:%M")

    st.title(f"🧠 {company} — Intelligence Dashboard")
    st.caption(f"Report generated · {generated_at}")

    pdf_bytes = generate_pdf(
        company, competitors, brief, news_out, comp_out,
        fin_out, improve_out, financials, ticker_map, generated_at,
    )
    st.download_button(
        label="📄 Download Full Report (PDF)",
        data=pdf_bytes,
        file_name=f"{company.replace(' ','_')}_Intelligence_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        type="primary",
    )

    st.divider()
    st.subheader("📋 Executive Brief")
    st.info(brief)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📰 News & Sentiment",
        "⚔️ Competitor Analysis",
        "💰 Financials",
        "🎯 What to Improve",
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Latest News — {company}")
            for article in company_news:
                st.markdown(f"**[{article['title']}]({article['link']})**")
                st.caption(f"{article['source']}  ·  {article['date']}")
                st.write(article["snippet"])
                st.markdown("---")
        with col2:
            st.subheader("AI News Analysis")
            st.markdown(news_out)

    with tab2:
        st.subheader("📈 Search Interest vs Competitors")
        trend_list = [trend_dfs.get(kw, pd.DataFrame()) for kw in all_keywords]
        if any(not df.empty for df in trend_list):
            st.plotly_chart(build_trend_chart(all_keywords, trend_list), width="stretch")
        if competitors:
            st.subheader("Competitor News")
            comp_tabs = st.tabs(competitors)
            for ctab, comp in zip(comp_tabs, competitors):
                with ctab:
                    for article in comp_news.get(comp, []):
                        st.markdown(f"**[{article['title']}]({article['link']})**")
                        st.caption(f"{article['source']}  ·  {article['date']}")
                        st.markdown("---")
        st.subheader("AI Competitor Analysis")
        st.markdown(comp_out)

    with tab3:
        if ticker_map:
            all_tickers = list(ticker_map.values())
            st.subheader("📈 Stock Price (Last 12 Months)")
            st.plotly_chart(build_stock_chart(all_tickers, stock_dfs), width="stretch")
            st.divider()
            render_financial_comparison(ticker_map, financials)
            st.divider()
            st.subheader("💵 Quarterly Revenue ($M)")
            st.plotly_chart(build_revenue_chart(all_tickers, financials), width="stretch")
            st.subheader("AI Financial Analysis")
            st.markdown(fin_out)
        else:
            st.warning("Could not resolve any stock tickers automatically.")

    with tab4:
        st.subheader(f"🎯 What {company} Should Improve")
        st.markdown(improve_out)


# ══════════════════════════════════════════════════════════════════════════════
# App entry point
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Company Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
)

if not st.session_state.get("logged_in"):
    show_login()
    st.stop()

settings = load_settings()

with st.sidebar:
    st.markdown(f"👤 **{st.session_state['user_id']}** "
                f"({'Admin' if is_admin() else 'Employee'})")
    if st.button("Logout", use_container_width=True):
        for key in ["logged_in", "user_id", "role", "company_settings"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()

    if is_admin():
        st.title("⚙️ Admin Settings")
        st.caption("Set once — employees see the report on open.")
        st.divider()

        company_name = st.text_input(
            "Company name",
            value=settings.get("company_name", ""),
            placeholder="e.g. Nike",
        )
        st.caption("Stock ticker is resolved automatically from the company name.")

        st.markdown("**Competitors**")
        auto_comp = st.checkbox(
            "Auto-detect competitors",
            value=settings.get("auto_competitors", True),
        )
        manual_competitors = st.text_input(
            "Or enter manually (comma-separated)",
            value=", ".join(settings.get("competitors", [])),
            placeholder="e.g. Adidas, Puma, New Balance",
            disabled=auto_comp,
        )

        if st.button("💾 Save & Load Dashboard", type="primary", use_container_width=True):
            if not company_name.strip():
                st.error("Please enter a company name.")
            else:
                competitors = []
                if auto_comp:
                    with st.spinner("Detecting competitors…"):
                        competitors = auto_detect_competitors(company_name.strip())
                    if competitors:
                        st.success(f"Detected: {', '.join(competitors)}")
                    else:
                        st.warning("Could not auto-detect. Enter manually.")
                else:
                    competitors = [c.strip() for c in manual_competitors.split(",") if c.strip()]

                save_settings({
                    "company_name":     company_name.strip(),
                    "company_ticker":   "",
                    "competitors":      competitors,
                    "auto_competitors": auto_comp,
                })
                st.rerun()

    if settings.get("company_name"):
        st.success(f"Tracking: **{settings['company_name']}**")
        if settings.get("competitors"):
            st.caption("vs " + ", ".join(settings["competitors"]))
        if st.button("🔄 Refresh Report", use_container_width=True):
            st.rerun()

if not settings.get("company_name"):
    st.title("🧠 Company Intelligence Dashboard")
    st.divider()
    if is_admin():
        st.info("👈 Open the sidebar, enter your company name, and click Save & Load Dashboard.")
    else:
        st.info("The dashboard is being configured by your admin. Please check back shortly.")
else:
    render_dashboard(settings)
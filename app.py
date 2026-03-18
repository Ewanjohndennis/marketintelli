import os, json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from serpapi import GoogleSearch
from huggingface_hub import InferenceClient
from datetime import datetime

load_dotenv()

HF_TOKEN     = st.secrets["HF_TOKEN"]
SERP_API_KEY = st.secrets["SERP_API_KEY"]
HF_MODEL     = "microsoft/Phi-3-mini-4k-instruct"


client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

COLORS = [
    ("#4F8EF7", "rgba(79,142,247,0.12)"),
    ("#F76A4F", "rgba(247,106,79,0.12)"),
    ("#2ECC71", "rgba(46,204,113,0.12)"),
    ("#F1C40F", "rgba(241,196,15,0.12)"),
    ("#9B59B6", "rgba(155,89,182,0.12)"),
]

# ── Ticker resolver ────────────────────────────────────────────────────────────
KNOWN_TICKERS = {
    "apple": "AAPL", "samsung": "005930.KS", "google": "GOOGL",
    "alphabet": "GOOGL", "microsoft": "MSFT", "amazon": "AMZN",
    "meta": "META", "facebook": "META", "tesla": "TSLA",
    "nvidia": "NVDA", "nike": "NKE", "adidas": "ADDYY",
    "puma": "PUMSY", "sony": "SONY", "lg": "066570.KS",
    "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM",
    "netflix": "NFLX", "spotify": "SPOT", "uber": "UBER",
    "airbnb": "ABNB", "coca cola": "KO", "pepsi": "PEP",
    "pepsico": "PEP", "walmart": "WMT", "target": "TGT",
    "disney": "DIS", "ford": "F", "gm": "GM", "toyota": "TM",
    "bmw": "BMWYY", "mercedes": "MBGYY", "volkswagen": "VWAGY",
    "pfizer": "PFE", "jpmorgan": "JPM", "goldman sachs": "GS",
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
    try:
        t = yf.Ticker(keyword.upper())
        if t.info.get("regularMarketPrice") or t.info.get("currentPrice"):
            return keyword.upper()
    except Exception:
        pass
    return None

# ── Agent prompts ──────────────────────────────────────────────────────────────
AGENTS = {
    "trend":
        "You are TrendAgent. Analyse the Google search trend data statistically. "
        "Identify momentum shifts, peaks, dips, and growth/decline signals. "
        "Be concise — 2 short paragraphs.",
    "sentiment":
        "You are SentimentAgent. Analyse the news headlines provided. "
        "Determine overall sentiment (positive/neutral/negative) for each entity. "
        "Highlight reputational risks or opportunities. 2 short paragraphs.",
    "competitor":
        "You are CompetitorAgent. Based on trend data and news headlines, "
        "compare relative positioning of the entities. "
        "Who is gaining, who is losing, and why? 2 short paragraphs.",
    "forecast":
        "You are ForecastAgent. Based on trend momentum and news signals, "
        "predict the next-quarter outlook for each entity. "
        "Be directional (e.g. 'up ~10%'). 2 short paragraphs.",
    "synthesizer":
        "You are the Synthesizer of a real-time market intelligence system. "
        "Merge all agent outputs into a structured final intelligence brief:\n"
        "1. Trend Summary\n2. Sentiment & Reputation\n"
        "3. Competitive Positioning\n4. Outlook & Recommendations\n"
        "Be sharp, actionable. Use bullet points within each section.",
    "sales_analyst":
        "You are SalesAnalyst. Given financial data for publicly listed companies, "
        "produce a structured sales intelligence brief:\n"
        "1. Revenue Trend\n2. Stock Performance & Market Sentiment\n"
        "3. Pricing Signal Analysis\n4. Market Share Estimate\n"
        "5. Sales Outlook for next quarter\n"
        "Be sharp and data-driven. Use bullet points.",
}

def call_agent(role: str, message: str) -> str:
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": AGENTS[role]},
            {"role": "user",   "content": message},
        ],
        max_tokens=600,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()

# ── Data fetchers ──────────────────────────────────────────────────────────────
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

def fetch_news(keyword: str, num: int = 5) -> list:
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
    try:
        df = yf.Ticker(ticker).history(period="1y")[["Close"]].reset_index()
        df.columns = ["date", "price"]
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df
    except Exception:
        return pd.DataFrame()

def fetch_financials(ticker: str) -> dict:
    try:
        t     = yf.Ticker(ticker)
        info  = t.info
        q_fin = t.quarterly_financials
        revenue_series = {}
        if q_fin is not None and not q_fin.empty:
            rev_row = q_fin.loc["Total Revenue"] if "Total Revenue" in q_fin.index else None
            if rev_row is not None:
                revenue_series = {
                    str(col.date()): int(val / 1e6)
                    for col, val in rev_row.items() if pd.notna(val)
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
        }
    except Exception:
        return {}

def fetch_pricing(keyword: str) -> list:
    try:
        data = GoogleSearch({
            "engine": "google_shopping", "q": keyword,
            "num": 5, "api_key": SERP_API_KEY,
        }).get_dict()
        return [
            {
                "title":  r.get("title", ""),
                "price":  r.get("price", "N/A"),
                "source": r.get("source", ""),
                "link":   r.get("link", ""),
            }
            for r in data.get("shopping_results", [])
        ]
    except Exception:
        return []

# ── Chart builders ─────────────────────────────────────────────────────────────
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
        hovermode="x unified", height=400,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(title="Interest (0–100)", gridcolor="rgba(180,180,180,0.15)"),
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
        hovermode="x unified", height=380,
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
        barmode="group", height=350,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(title="Revenue ($M)", gridcolor="rgba(180,180,180,0.15)"),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# App UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Real-Time Market Intelligence", page_icon="🧠", layout="wide")
st.title("🧠 Real-Time Market Intelligence")
st.caption("Track brands, products, and markets — powered by Google Trends, live news, financial data, and AI agents.")
st.divider()

track_type = st.radio(
    "What are you tracking?",
    ["Brand", "Product", "Market / Category", "Custom"],
    horizontal=True,
)
placeholders = {
    "Brand":             "e.g. Apple, Samsung, Sony",
    "Product":           "e.g. iPhone 15, Galaxy S24",
    "Market / Category": "e.g. Electric Vehicles, Cloud Computing",
    "Custom":            "Enter any keywords",
}
raw_input  = st.text_input(f"Enter keywords ({track_type})", placeholder=placeholders[track_type])
news_count = st.slider("News articles per keyword", 3, 10, 5)

if st.button("🚀 Run Full Intelligence Report", type="primary"):
    keywords = [k.strip() for k in raw_input.split(",") if k.strip()]
    if not keywords:
        st.warning("Please enter at least one keyword.")
        st.stop()
    if len(keywords) > 5:
        st.warning("Please enter at most 5 keywords.")
        st.stop()

    # ── Step 1: Resolve tickers ────────────────────────────────────────────
    with st.spinner("Resolving stock tickers…"):
        ticker_map = {}
        for kw in keywords:
            t = resolve_ticker(kw)
            if t:
                ticker_map[kw] = t

    if ticker_map:
        st.success("Tickers resolved: " +
                   ", ".join([f"**{kw}** → `{t}`" for kw, t in ticker_map.items()]))

    # ── Step 2: Fetch ALL data in parallel ────────────────────────────────
    with st.spinner("⚡ Fetching all data in parallel…"):
        with ThreadPoolExecutor(max_workers=12) as ex:
            # Submit all fetches, keyed by index/name for safe retrieval
            trend_futures   = [(kw, ex.submit(fetch_trend, kw))   for kw in keywords]
            news_futures    = [(kw, ex.submit(fetch_news, kw, news_count)) for kw in keywords]
            stock_futures   = [(t,  ex.submit(fetch_stock_price, t)) for t in ticker_map.values()]
            fin_futures     = [(t,  ex.submit(fetch_financials, t))  for t in ticker_map.values()]
            pricing_futures = [(kw, ex.submit(fetch_pricing, kw))  for kw in keywords]

            # Collect results safely — no DataFrames used as dict keys
            dataframes_list = [f.result() for _, f in trend_futures]
            news_data       = {kw: f.result() for kw, f in news_futures}
            stock_dfs       = {t:  f.result() for t,  f in stock_futures}
            financials      = {t:  f.result() for t,  f in fin_futures}
            pricing         = {kw: f.result() for kw, f in pricing_futures}

    # ── Step 3: Build agent inputs ─────────────────────────────────────────
    trend_lines = [
        f"{kw}: avg={df['value'].mean():.1f}, peak={df['value'].max()}, "
        f"series={df['value'].astype(int).tolist()}"
        for kw, df in zip(keywords, dataframes_list) if not df.empty
    ]
    news_lines = [
        f"{kw}: {' | '.join([a['title'] for a in articles[:5]])}"
        for kw, articles in news_data.items()
    ]
    trend_summary = "\n".join(trend_lines)
    news_summary  = "\n".join(news_lines)
    combined      = f"TREND DATA:\n{trend_summary}\n\nNEWS HEADLINES:\n{news_summary}"

    fin_lines = []
    for kw, ticker in ticker_map.items():
        fin   = financials.get(ticker, {})
        rev_q = fin.get("revenue_quarterly", {})
        rev_str = ", ".join([f"{d}: ${v}M" for d, v in sorted(rev_q.items())])
        fin_lines.append(
            f"{fin.get('company_name', kw)} ({ticker}):\n"
            f"  Market Cap: ${fin.get('market_cap',0)/1e9:.1f}B\n"
            f"  TTM Revenue: ${fin.get('revenue_ttm',0)/1e9:.1f}B\n"
            f"  Gross Margin: {fin.get('gross_margin',0)*100:.1f}%\n"
            f"  P/E Ratio: {fin.get('pe_ratio',0):.1f}\n"
            f"  52W High/Low: ${fin.get('52w_high',0):.2f} / ${fin.get('52w_low',0):.2f}\n"
            f"  Current Price: ${fin.get('current_price',0):.2f}\n"
            f"  Quarterly Revenue: {rev_str}"
        )
    pricing_lines = [
        f"{kw}: {', '.join([p['price'] for p in items if p['price'] != 'N/A'][:5])}"
        for kw, items in pricing.items()
    ]
    sales_input = (
        "FINANCIAL DATA:\n" + "\n\n".join(fin_lines) +
        "\n\nPRICING SIGNALS:\n" + "\n".join(pricing_lines)
    )

    # ── Step 4: Run ALL agents in parallel ────────────────────────────────
    with st.spinner("🤖 Running all AI agents in parallel…"):
        with ThreadPoolExecutor(max_workers=5) as ex:
            f_trend      = ex.submit(call_agent, "trend",         f"Analyse trends.\n\n{trend_summary}")
            f_sentiment  = ex.submit(call_agent, "sentiment",     f"Analyse sentiment.\n\n{news_summary}")
            f_competitor = ex.submit(call_agent, "competitor",    f"Compare positioning.\n\n{combined}")
            f_forecast   = ex.submit(call_agent, "forecast",      f"Forecast outlook.\n\n{combined}")
            f_sales      = ex.submit(call_agent, "sales_analyst", sales_input)

            trend_out      = f_trend.result()
            sentiment_out  = f_sentiment.result()
            competitor_out = f_competitor.result()
            forecast_out   = f_forecast.result()
            sales_brief    = f_sales.result()

    # Synthesizer runs last — needs the 4 agent outputs above
    with st.spinner("🧠 Synthesizing final brief…"):
        brief = call_agent(
            "synthesizer",
            f"TrendAgent:\n{trend_out}\n\nSentimentAgent:\n{sentiment_out}\n\n"
            f"CompetitorAgent:\n{competitor_out}\n\nForecastAgent:\n{forecast_out}",
        )

    # ── Step 5: Render both tabs with all results ready ────────────────────
    tab_trend, tab_sales = st.tabs(["📊 Trend & News Intelligence", "💰 Sales Intelligence"])

    with tab_trend:
        st.subheader("📈 Search Interest (Last 12 Months)")
        if any(not df.empty for df in dataframes_list):
            st.plotly_chart(build_trend_chart(keywords, dataframes_list), use_container_width=True)
            cols = st.columns(len(keywords) * 2)
            for i, (kw, df) in enumerate(zip(keywords, dataframes_list)):
                if not df.empty:
                    cols[i * 2].metric(f"{kw} avg",  int(df["value"].mean()))
                    cols[i * 2 + 1].metric(f"{kw} peak", int(df["value"].max()))
        else:
            st.error("No trend data returned. Check your SERP_API_KEY.")

        st.divider()

        st.subheader("📰 Latest News")
        news_tabs = st.tabs(keywords)
        for tab, kw in zip(news_tabs, keywords):
            with tab:
                articles = news_data.get(kw, [])
                if not articles:
                    st.info("No news articles found.")
                for article in articles:
                    st.markdown(f"**[{article['title']}]({article['link']})**")
                    st.caption(f"{article['source']}  ·  {article['date']}")
                    st.write(article["snippet"])
                    st.markdown("---")

        st.divider()

        st.subheader("🤖 Agent Outputs")
        with st.expander("📈 TrendAgent",      expanded=False): st.markdown(trend_out)
        with st.expander("💬 SentimentAgent",  expanded=False): st.markdown(sentiment_out)
        with st.expander("⚔️ CompetitorAgent", expanded=False): st.markdown(competitor_out)
        with st.expander("🔮 ForecastAgent",   expanded=False): st.markdown(forecast_out)

        st.subheader("📋 Intelligence Brief")
        st.caption(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.info(brief)

    with tab_sales:
        if not ticker_map:
            st.warning(
                "Could not resolve stock tickers. "
                "Try publicly listed company names like 'Apple', 'Samsung', 'Nike'."
            )
        else:
            resolved_tickers  = list(ticker_map.values())
            resolved_keywords = list(ticker_map.keys())

            st.subheader("📈 Stock Price (Last 12 Months)")
            st.plotly_chart(build_stock_chart(resolved_tickers, stock_dfs), use_container_width=True)

            st.subheader("🏦 Key Financial Metrics")
            metric_cols = st.columns(len(resolved_tickers))
            for col, (kw, ticker) in zip(metric_cols, ticker_map.items()):
                fin = financials.get(ticker, {})
                if not fin:
                    col.warning(f"No data for {ticker}")
                    continue
                col.markdown(f"**{fin.get('company_name', kw)}** (`{ticker}`)")
                col.metric("Current Price", f"${fin.get('current_price', 0):.2f}")
                col.metric("Market Cap",    f"${fin.get('market_cap', 0)/1e9:.1f}B")
                col.metric("TTM Revenue",   f"${fin.get('revenue_ttm', 0)/1e9:.1f}B")
                col.metric("Gross Margin",  f"{fin.get('gross_margin', 0)*100:.1f}%")
                col.metric("P/E Ratio",     f"{fin.get('pe_ratio', 0):.1f}")
                col.metric("52W High",      f"${fin.get('52w_high', 0):.2f}")
                col.metric("52W Low",       f"${fin.get('52w_low', 0):.2f}")

            st.divider()

            st.subheader("💵 Quarterly Revenue Comparison ($M)")
            st.plotly_chart(build_revenue_chart(resolved_tickers, financials), use_container_width=True)

            st.divider()

            st.subheader("🏷️ Pricing Signals (Google Shopping)")
            price_tabs = st.tabs(resolved_keywords)
            for ptab, kw in zip(price_tabs, resolved_keywords):
                with ptab:
                    items = pricing.get(kw, [])
                    if not items:
                        st.info("No pricing data found.")
                    for item in items:
                        st.markdown(f"**{item['title']}** — `{item['price']}`")
                        st.caption(item["source"])
                        st.markdown("---")

            st.divider()

            st.subheader("📋 Sales Intelligence Brief")
            st.caption(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(sales_brief)
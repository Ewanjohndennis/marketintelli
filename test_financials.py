import yfinance as yf

TICKER = "AAPL"

print("=" * 60)
print(f"Testing yfinance for ticker: {TICKER}")
print("=" * 60)

t = yf.Ticker(TICKER)

# 1. Basic info
print("\n1. Company Info:")
try:
    info = t.info
    print(f"   Company:      {info.get('longName')}")
    print(f"   Price:        {info.get('currentPrice')}")
    print(f"   Market Cap:   {info.get('marketCap')}")
    print(f"   P/E Ratio:    {info.get('trailingPE')}")
    print(f"   EPS:          {info.get('trailingEps')}")
    print(f"   Gross Margin: {info.get('grossMargins')}")
    print(f"   Op Margin:    {info.get('operatingMargins')}")
    print(f"   ROE:          {info.get('returnOnEquity')}")
    print(f"   D/E Ratio:    {info.get('debtToEquity')}")
    print(f"   Div Yield:    {info.get('dividendYield')}")
    print(f"   52W High:     {info.get('fiftyTwoWeekHigh')}")
    print(f"   52W Low:      {info.get('fiftyTwoWeekLow')}")
    print(f"   TTM Revenue:  {info.get('totalRevenue')}")
except Exception as e:
    print(f"   ERROR: {e}")

# 2. Quarterly financials
print("\n2. Quarterly Revenue:")
try:
    q_fin = t.quarterly_financials
    if q_fin is not None and not q_fin.empty:
        if "Total Revenue" in q_fin.index:
            rev_row = q_fin.loc["Total Revenue"]
            for col, val in rev_row.items():
                if str(val) != "nan":
                    print(f"   {str(col.date())}: ${int(val)/1e9:.2f}B")
        else:
            print(f"   'Total Revenue' not found. Available rows: {list(q_fin.index[:5])}")
    else:
        print("   Empty quarterly financials")
except Exception as e:
    print(f"   ERROR: {e}")

# 3. Stock price history
print("\n3. Stock Price (last 5 days):")
try:
    hist = t.history(period="5d")[["Close"]]
    for date, row in hist.iterrows():
        print(f"   {str(date.date())}: ${row['Close']:.2f}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\nDone.")
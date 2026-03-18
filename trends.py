from pytrends.request import TrendReq
import time
pytrends = TrendReq(hl='en-US', tz=330)
kw_list=["nike shoes", "Bata shoes", "Puma Shoes"]
time.sleep(60)
pytrends.build_payload(kw_list)

data = pytrends.interest_over_time()
print(data)
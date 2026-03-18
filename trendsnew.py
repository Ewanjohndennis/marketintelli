import os
from dotenv import load_dotenv

from serpapi import GoogleSearch
import pandas as pd
import streamlit as st

load_dotenv()
hf = os.getenv("SERP_API_KEY")

kw_list = ["Nike Shoes", "Bata Shoes", "Puma Shoes"]

for keyword in kw_list:

    params = {
        "q": keyword,
        "engine": "google",
        "location": "Austin, Texas",
        "hl": "en",
        "gl": "us",
        "num": 5,
        "api_key": hf
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic = results.get("organic_results", [])

    print(f"\nResults for: {keyword}\n")

    for r in organic[:5]:
        print("Title:", r.get("title"))
        print("Link:", r.get("link"))
        print("Snippet:", r.get("snippet"))
        print("-" * 50)
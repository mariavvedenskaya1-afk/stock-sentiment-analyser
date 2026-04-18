import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

# Load FinBERT sentiment model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Fetch news from NewsAPI
def fetch_news(ticker, api_key):
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [{"headline": a["title"], "source": a["source"]["name"], "date": a["publishedAt"][:10]} for a in articles if a["title"]]

# Main app
st.set_page_config(page_title="Stock Sentiment Analyser", page_icon="📈")
st.title("📈 Stock Sentiment Analyser")
st.caption("Powered by FinBERT — an AI model trained on financial text")

api_key = st.text_input("Enter your NewsAPI key", type="password")
ticker = st.text_input("Enter a company name or ticker (e.g. Apple, Tesla, NVIDIA)")

if st.button("Analyse Sentiment") and ticker and api_key:
    with st.spinner("Fetching news and analysing sentiment..."):
        classifier = load_model()
        articles = fetch_news(ticker, api_key)

        if not articles:
            st.error("No articles found. Check your API key or try a different company name.")
        else:
            headlines = [a["headline"] for a in articles]
            results = classifier(headlines, truncation=True, max_length=512)

            for i, article in enumerate(articles):
                article["sentiment"] = results[i]["label"]
                article["confidence"] = round(results[i]["score"] * 100, 1)

            df = pd.DataFrame(articles)

            # Summary metrics
            counts = df["sentiment"].value_counts()
            pos = counts.get("positive", 0)
            neg = counts.get("negative", 0)
            neu = counts.get("neutral", 0)
            score = round((pos - neg) / len(df) * 100)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Score", f"{score:+}%")
            col2.metric("Positive", pos)
            col3.metric("Neutral", neu)
            col4.metric("Negative", neg)

            # Colour-coded table
            def colour_sentiment(val):
                if val == "positive": return "background-color: #d4edda; color: #155724"
                if val == "negative": return "background-color: #f8d7da; color: #721c24"
                return "background-color: #fff3cd; color: #856404"

            st.subheader(f"Latest headlines for '{ticker}'")
            styled = df[["date", "source", "headline", "sentiment", "confidence"]].style.map(colour_sentiment, subset=["sentiment"])
            st.dataframe(styled, use_container_width=True)

            st.bar_chart(counts)
        
import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def fetch_news(ticker, api_key):
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [{"headline": a["title"], "source": a["source"]["name"], "date": a["publishedAt"][:10], "url": a["url"]} for a in articles if a["title"]]

st.set_page_config(page_title="Stock Sentiment Analyser", page_icon="⭐️")
st.markdown("""
<style>
.stApp {
    background-image: url("https://www.religareonline.com/blog/wp-content/uploads/2023/05/online-trading.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-color: rgba(0,0,0,0.6);
}
</style>
""", unsafe_allow_html=True)
st.title("⭐️ Stock Sentiment Analyser")
st.caption("Powered by FinBERT — an AI model trained on financial text")


api_key = st.secrets["NEWSAPI_KEY"]
ticker = st.text_input("Enter a company name or ticker (e.g. Apple, Tesla, NVIDIA)")

if st.button("Analyse Sentiment") and ticker:
    with st.spinner("Fetching news and analysing sentiment..."):
        classifier = load_model()
        articles = fetch_news(ticker, api_key)

        if not articles:
            st.error("No articles found. Try a different company name.")
        else:
            headlines = [a["headline"] for a in articles]
            results = classifier(headlines, truncation=True, max_length=512)

            for i, article in enumerate(articles):
                article["sentiment"] = results[i]["label"]
                article["confidence"] = round(results[i]["score"] * 100, 1)

            df = pd.DataFrame(articles)
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

            st.subheader(f"Latest headlines for '{ticker}'")
            st.markdown("🟢 Positive &nbsp;&nbsp; 🟡 Neutral &nbsp;&nbsp; 🔴 Negative")

            for article in articles:
                sentiment = article["sentiment"]
                if sentiment == "positive":
                    colour = "🟢"
                elif sentiment == "negative":
                    colour = "🔴"
                else:
                    colour = "🟡"

                st.markdown(f"{colour} **[{article['headline']}]({article['url']})** — *{article['source']}* · {article['date']}")
                st.caption(f"Sentiment: {article['sentiment']} ({article['confidence']}% confidence)")
                st.divider()

            st.bar_chart(counts)

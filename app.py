import streamlit as st
import requests
import pandas as pd
import altair as alt
from transformers import pipeline
from datetime import datetime, timedelta

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def fetch_news(ticker, api_key):
    one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=relevancy&pageSize=20&from={one_month_ago}&excludeDomains=spinalcolumnradiology.com,pinkvilla.com,tmz.com&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [{"headline": a["title"], "source": a["source"]["name"], "date": a["publishedAt"][:10], "url": a["url"]} for a in articles if a["title"]]

st.set_page_config(page_title="Stock Sentiment Analyser", page_icon="⭐️")

st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
    url("https://www.religareonline.com/blog/wp-content/uploads/2023/05/online-trading.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
input:focus {
    border-color: #9af540 !important;
    box-shadow: 0 0 0 2px #9af540 !important;
    outline: none !important;
}
[data-baseweb="input"]:focus-within {
    border-color: #9af540 !important;
    box-shadow: 0 0 0 2px #9af540 !important;
}
* {
    color: white !important;
}
.st-emotion-cache-1wbqy5l {
    color: white !important;
}
p, h1, h2, h3, label, span {
    color: white !important;
}
button {
    background-color: black !important;
    color: white !important;
    border: 1px solid black !important;
}
button p {
    color: white !important;
}

input {
    background-color: #262731 !important;
    color: white !important;
    border: 1px solid #262731 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Stock Sentiment Analyser")
st.caption("Powered by FinBERT – an AI model trained on financial text")

api_key = st.secrets["NEWSAPI_KEY"]
ticker = st.text_input("Enter a company name or ticker (e.g. Apple, Tesla, NVIDIA)")

if st.button("Analyse Sentiment") and ticker:
    with st.spinner("Fetching news and analysing sentiment..."):
        classifier = load_model()
        articles = fetch_news(ticker, api_key)
        articles = sorted(articles, key=lambda x: x["date"], reverse=True)

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
            dates = [a["date"] for a in articles]
            min_date = datetime.strptime(min(dates), "%Y-%m-%d").strftime("%d/%m/%Y")
            max_date = datetime.strptime(max(dates), "%Y-%m-%d").strftime("%d/%m/%Y")
            st.caption(f"Showing articles from {min_date} to {max_date} (last 30 days)")
            st.markdown("🤩 Positive  \n😶 Neutral  \n🫪 Negative")

            for article in articles:
                sentiment = article["sentiment"]
                if sentiment == "positive":
                    colour = "🤩"
                elif sentiment == "negative":
                    colour = "🫪"
                else:
                    colour = "😶"

                date_uk = datetime.strptime(article["date"], "%Y-%m-%d").strftime("%d/%m/%Y")
                st.markdown(f"{colour} **[{article['headline']}]({article['url']})** – *{article['source']}* · {date_uk}")
                st.caption(f"Sentiment: {article['sentiment']} ({article['confidence']}% confidence)")
                st.divider()

            counts_df = counts.reset_index()
            counts_df.columns = ["sentiment", "count"]
            chart = alt.Chart(counts_df).mark_bar(color="#be26ff").encode(
                x=alt.X("sentiment", title="Sentiment"),
                y=alt.Y("count", title="Count")
            )
            st.altair_chart(chart, use_container_width=True)

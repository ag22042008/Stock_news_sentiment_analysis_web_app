import streamlit as st
import pymongo
import finnhub
import pandas as pd
from transformers import pipeline
import datetime
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Multi-Stock Sentiment Analyzer", page_icon="📊", layout="wide")

st.title("📊 Multi-Company Strategic Sentiment Dashboard")

@st.cache_resource
def init_mongodb():
    try:
        client = pymongo.MongoClient(st.secrets["mongo"]["uri"], serverSelectionTimeoutMS=5000)
        return client["stock_analysis_db"]["news_articles"]
    except Exception as e:
        st.error(f"MongoDB Error: {e}")
        return None

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

news_collection = init_mongodb()
sentiment_pipeline = load_sentiment_model()

def fetch_and_store_news(tickers, start_date, end_date, api_key):
    finnhub_client = finnhub.Client(api_key=api_key)
    delta = end_date - start_date
    total_days = delta.days + 1
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(tickers):
        for d in range(total_days):
            date_str = (start_date + datetime.timedelta(days=d)).strftime("%Y-%m-%d")
            try:
                news_list = finnhub_client.company_news(ticker, _from=date_str, to=date_str)
                for item in news_list:
                    unique_id = f"{ticker}_{item.get('id')}"
                    doc = {
                        "_id": unique_id, "ticker": ticker.upper(), "headline": item.get("headline", ""),
                        "source": item.get("source", ""), "date_string": date_str
                    }
                    news_collection.update_one({"_id": unique_id}, {"$set": doc}, upsert=True)
            except: continue
            progress_bar.progress(((i * total_days) + d + 1) / (len(tickers) * total_days))
            time.sleep(1.1)

def get_news_from_mongo(tickers, start_date, end_date):
    query = {
        "ticker": {"$in": [t.upper() for t in tickers]},
        "date_string": {"$gte": start_date.strftime("%Y-%m-%d"), "$lte": end_date.strftime("%Y-%m-%d")}
    }
    return list(news_collection.find(query, {"_id": 0}))

def process_sentiment(df):
    results = sentiment_pipeline(df["headline"].tolist())
    df["sentiment_label"] = [res["label"] for res in results]
    df["sentiment_score"] = [res["score"] for res in results]
    
    label_map = {"positive": 1, "negative": -1, "neutral": 0}
    df["numeric_sentiment"] = df["sentiment_label"].map(label_map)
    return df

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Finnhub API Key", type="password")
    tickers = st.multiselect("Select Companies", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"], default=["AAPL", "MSFT"])
    start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=7))
    end_date = st.date_input("End Date", datetime.date.today())
    analyze_button = st.button("🚀 Run Analysis")

if analyze_button and api_key:
    fetch_and_store_news(tickers, start_date, end_date, api_key)
    raw_news = get_news_from_mongo(tickers, start_date, end_date)
    
    if raw_news:
        df = process_sentiment(pd.DataFrame(raw_news))
        
        company_stats = df.groupby("ticker").agg(
            overall_sentiment_score=("numeric_sentiment", "mean"),
            majority_sentiment=("sentiment_label", lambda x: x.mode()[0])
        ).reset_index()

        st.subheader("📈 Comparative Company Analysis")
        st.dataframe(company_stats, use_container_width=True)
        
        st.subheader("📊 Sentiment Distribution & Scoring")
        col_charts = st.columns(len(company_stats))
        
        for idx, row in company_stats.iterrows():
            with col_charts[idx]:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = row['overall_sentiment_score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': row['ticker']},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-1, -0.2], 'color': "red"},
                            {'range': [-0.2, 0.2], 'color': "gray"},
                            {'range': [0.2, 1], 'color': "green'"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Detailed Article Breakdown")
        st.dataframe(df[["ticker", "headline", "sentiment_label", "sentiment_score"]], use_container_width=True)
    else:
        st.warning("No data found for selected companies.")

import streamlit as st
import pymongo
import finnhub
import pandas as pd
from transformers import pipeline
import datetime
import time

st.set_page_config(page_title="Stock News Sentiment Analyzer", page_icon="📊", layout="wide")

st.title("📊 Strategic Stock News Sentiment Analyzer")
st.markdown("Fetch real-time financial market news, preserve history in MongoDB, and evaluate market sentiment with FinBERT.")

@st.cache_resource
def init_mongodb():
    try:
        mongo_uri = st.secrets["mongo"]["uri"]
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client["stock_analysis_db"]
        collection = db["news_articles"]
        return collection
    except Exception as e:
        st.error(f"Could not connect to MongoDB. Please check your Streamlit secrets! Error: {e}")
        return None

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

news_collection = init_mongodb()
sentiment_pipeline = load_sentiment_model()

def fetch_and_store_news(ticker, start_date, end_date, api_key):
    if news_collection is None:
        st.error("MongoDB connection is unavailable. Cannot store records.")
        return

    finnhub_client = finnhub.Client(api_key=api_key)
    
    delta = end_date - start_date
    total_days = delta.days + 1
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    saved_count = 0

    for i in range(total_days):
        current_day = start_date + datetime.timedelta(days=i)
        date_str = current_day.strftime("%Y-%m-%d")
        
        status_text.text(f"Fetching data from Finnhub for {date_str}...")
        
        try:
            news_list = finnhub_client.company_news(ticker, _from=date_str, to=date_str)
            
            day_saved = 0
            for item in news_list:
                unique_id = f"{ticker}_{item.get('id')}"
                
                document = {
                    "_id": unique_id,
                    "ticker": ticker.upper(),
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "datetime": datetime.datetime.fromtimestamp(item.get("datetime", time.time())),
                    "date_string": date_str
                }
                
                news_collection.update_one({"_id": unique_id}, {"$set": document}, upsert=True)
                day_saved += 1
            
            saved_count += day_saved
            
        except Exception as e:
            st.warning(f"Failed to process records for {date_str}. Error details: {str(e)}")
        
        progress_bar.progress((i + 1) / total_days)
        time.sleep(1.1)
        
    status_text.empty()
    progress_bar.empty()
    st.success(f"Successfully processed and synchronized {saved_count} articles into MongoDB!")

def get_news_from_mongo(ticker, start_date, end_date):
    if news_collection is None:
        return []
        
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    query = {
        "ticker": ticker.upper(),
        "date_string": {"$gte": start_str, "$lte": end_str}
    }
    
    return list(news_collection.find(query, {"_id": 0}))

def process_sentiment(df):
    with st.spinner("Analyzing headline dynamics using FinBERT..."):
        headlines = df["headline"].tolist()
        results = sentiment_pipeline(headlines)
        
        labels = [res["label"] for res in results]
        scores = [res["score"] for res in results]
        
        df["sentiment_label"] = labels
        df["sentiment_score"] = scores
    return df

with st.sidebar:
    st.header("⚙️ Configuration Settings")
    finnhub_api_key = st.text_input("Finnhub API Key", type="password")
    st.markdown("---")
    ticker_input = st.text_input("Stock Ticker Symbol", value="AAPL").upper().strip()
    
    default_start = datetime.date.today() - datetime.timedelta(days=7)
    default_end = datetime.date.today()
    
    start_date_input = st.date_input("Start Date", value=default_start)
    end_date_input = st.date_input("End Date", value=default_end)
    analyze_button = st.button("🚀 Run Complete Pipeline", use_container_width=True)

if analyze_button:
    if not finnhub_api_key:
        st.error("Please enter your Finnhub API Key in the sidebar configuration section.")
    elif start_date_input > end_date_input:
        st.error("Configuration Error: Start Date cannot fall after the specified End Date.")
    else:
        st.info(f"Step 1: Contacting Finnhub to fetch and sync data for **{ticker_input}**...")
        fetch_and_store_news(ticker_input, start_date_input, end_date_input, finnhub_api_key)
        
        st.info("Step 2: Retrieving compiled dataset from your MongoDB Atlas Cluster...")
        raw_news = get_news_from_mongo(ticker_input, start_date_input, end_date_input)
        
        if not raw_news:
            st.warning(f"No matching news items found inside the database for {ticker_input} within the selected timeframe.")
        else:
            df_news = pd.DataFrame(raw_news)
            st.success(f"Step 3: Found {len(df_news)} compiled historical records. Running FinBERT model...")
            
            df_processed = process_sentiment(df_news)
            
            st.markdown("---")
            st.subheader(f"📊 Market Sentiment Dashboard for {ticker_input}")
            
            col1, col2, col3 = st.columns(3)
            counts = df_processed["sentiment_label"].value_counts()
            
            col1.metric("Positive Articles", counts.get("positive", 0))
            col2.metric("Neutral Articles", counts.get("neutral", 0))
            col3.metric("Negative Articles", counts.get("negative", 0))
            
            st.markdown("### 📈 Sentiment Distribution Breakdown")
            st.bar_chart(counts)
            
            st.markdown("### 📋 Document Source Records & Analysis Insights")
            st.dataframe(
                df_processed[["datetime", "source", "headline", "sentiment_label", "sentiment_score"]],
                use_container_width=True
            )

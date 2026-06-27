import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from transformers import pipeline
import plotly.express as px
import pymongo

st.set_page_config(page_title="Stock Sentiment Analyzer", page_icon="📈", layout="wide")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_resource
def init_mongo():
    client = pymongo.MongoClient(st.secrets["mongo"]["uri"])
    return client

try:
    client = init_mongo()
    db = client.financial_db
    news_collection = db.news
except Exception as e:
    st.error("Could not connect to MongoDB. Please check your Streamlit secrets!")

def fetch_and_store_news(ticker, start_date, end_date, api_key):
    current_date = start_date
    progress_text = "Fetching and storing news to Database..."
    progress_bar = st.progress(0, text=progress_text)
    total_days = (end_date - start_date).days + 1
    days_processed = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from={date_str}&to={date_str}&token={api_key}'
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                daily_news = response.json()
                for article in daily_news:
                    news_collection.update_one(
                        {"id": article["id"]}, 
                        {"$set": article}, 
                        upsert=True
                    )
        except Exception as e:
            st.warning(f"Failed to fetch data for {date_str}.")
            
        current_date += timedelta(days=1)
        days_processed += 1
        progress_bar.progress(days_processed / total_days, text=f"Syncing data for {date_str} to MongoDB...")
        time.sleep(0.5) 
        
    progress_bar.empty()

def get_news_from_mongo(ticker, start_date, end_date):
    start_unix = int(start_date.timestamp())
    end_unix = int((end_date + timedelta(days=1)).timestamp())
    
    query = {
        "related": ticker,
        "datetime": {"$gte": start_unix, "$lt": end_unix}
    }
    
    return list(news_collection.find(query, {"_id": 0}))

def clean_news_data(raw_news):
    df = pd.DataFrame(raw_news)
    if df.empty:
        return df
        
    columns_to_keep = ['datetime', 'related', 'source', 'headline', 'summary']
    df_clean = df[columns_to_keep].copy()
    
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'], unit='s')
    df_clean['news'] = df_clean['headline'] + ". " + df_clean['summary']
    df_clean = df_clean.drop(columns=['headline', 'summary'])
    df_clean = df_clean.sort_values(by='datetime', ascending=False).reset_index(drop=True)
    return df_clean

def analyze_sentiment(df, sentiment_analyzer):
    def get_sentiment_score(text):
        try:
            result = sentiment_analyzer(text, truncation=True, max_length=512)[0]
            label = result['label']
            confidence = result['score']
            
            if label == 'positive': multiplier = 1
            elif label == 'negative': multiplier = -1
            else: multiplier = 0
                
            numeric_score = multiplier * confidence
            return pd.Series([label, confidence, numeric_score])
        except:
            return pd.Series(["neutral", 0.0, 0.0])

    df[['sentiment_label', 'confidence', 'numeric_score']] = df['news'].apply(get_sentiment_score)
    return df

def calculate_overall_sentiment(df):
    overall_scores = df.groupby('related').agg(
        overall_sentiment_score=('numeric_score', 'mean'), 
        total_articles=('numeric_score', 'count'),
        majority_sentiment=('sentiment_label', lambda x: x.mode()[0] if not x.mode().empty else "neutral")
    ).reset_index()
    
    return overall_scores.sort_values(by='overall_sentiment_score', ascending=False)

def plot_sentiment(df_overall):
    fig = px.bar(
        df_overall, 
        x='related', 
        y='overall_sentiment_score', 
        color='overall_sentiment_score',
        color_continuous_scale=['#ff4b4b', '#f9f9f9', '#09ab3b'], 
        range_color=[-1, 1],
        text_auto='.2f',
        labels={'overall_sentiment_score': 'Sentiment Score', 'related': 'Company Ticker'}
    )
    
    fig.update_layout(
        title=dict(text='Overall News Sentiment by Company', font=dict(size=18, color='white')),
        xaxis=dict(showgrid=False, title=''),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='gray', zerolinewidth=2),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def style_dataframe(df):
    def highlight_sentiment(val):
        if val == 'positive': return 'color: #09ab3b; font-weight: bold;'
        elif val == 'negative': return 'color: #ff4b4b; font-weight: bold;'
        return 'color: gray;'
    return df.style.map(highlight_sentiment, subset=['sentiment_label'])

st.title("📊 ML Financial News Sentiment Analyzer")
st.markdown("""
    <div style='background-color: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 5px solid #4facfe; margin-bottom: 20px;'>
        <h4 style='margin:0; color: white;'>Uncover Market Sentiment in Seconds 🚀</h4>
        <p style='margin:0; color: #a6accd;'>Powered by <b>FinBERT AI</b> and <b>MongoDB</b> to analyze and store historical market sentiment.</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuration")
api_key_input = st.sidebar.text_input("🔑 Finnhub API Key", type="password", help="Get a free key from finnhub.io")
ticker_input = st.sidebar.text_input("📈 Stock Ticker (e.g., AAPL, MSFT)", "AAPL").upper()

today = datetime.now()
seven_days_ago = today - timedelta(days=7)
start_date = st.sidebar.date_input("📅 Start Date", seven_days_ago)
end_date = st.sidebar.date_input("📅 End Date", today)

st.sidebar.divider()
analyze_button = st.sidebar.button("🚀 Analyze Sentiment", use_container_width=True, type="primary")

if analyze_button:
    if not api_key_input:
        st.error("⚠️ Please enter your Finnhub API Key in the sidebar.")
    elif start_date > end_date:
        st.error("⚠️ Start Date cannot be after End Date.")
    else:
        with st.spinner("🤖 Loading FinBERT AI Model..."):
            analyzer = load_model()
            
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        fetch_and_store_news(ticker_input, start_datetime, end_datetime, api_key_input)
        
        raw_news = get_news_from_mongo(ticker_input, start_datetime, end_datetime)

import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from transformers import pipeline
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Stock Sentiment Analyzer", page_icon="📈", layout="wide")

# --- 2. LOAD AI MODEL (Cached so it only downloads once) ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# --- 3. OUR BACKEND FUNCTIONS ---
def fetch_stock_news(ticker, start_date, end_date, api_key):
    """Fetches news day by day to bypass Finnhub limits."""
    all_news = []
    current_date = start_date
    
    # Create a progress bar in the app
    progress_text = "Fetching daily news..."
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
                all_news.extend(daily_news)
        except Exception as e:
            st.warning(f"Failed to fetch data for {date_str}.")
            
        current_date += timedelta(days=1)
        days_processed += 1
        progress_bar.progress(days_processed / total_days, text=f"Fetching data for {date_str}...")
        time.sleep(0.5) # Pause to avoid spamming the API
        
    progress_bar.empty() # Remove progress bar when done
    return all_news

def clean_news_data(raw_news):
    df = pd.DataFrame(raw_news)
    if df.empty:
        return df
        
    columns_to_keep = ['datetime', 'related', 'source', 'headline', 'summary']
    df_clean = df[columns_to_keep].copy()
    
    # Fix datetime
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'], unit='s')
    
    # Combine text
    df_clean['news'] = df_clean['headline'] + ". " + df_clean['summary']
    df_clean = df_clean.drop(columns=['headline', 'summary'])
    
    # Sort and reset index
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
    # Create an interactive, sleek Plotly chart
    fig = px.bar(
        df_overall, 
        x='related', 
        y='overall_sentiment_score', 
        color='overall_sentiment_score',
        color_continuous_scale=['#ff4b4b', '#f9f9f9', '#09ab3b'], # Red to Green
        range_color=[-1, 1],
        text_auto='.2f',
        labels={'overall_sentiment_score': 'Sentiment Score', 'related': 'Company Ticker'}
    )
    
    # Style the background and grid lines
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
    """Highlights the sentiment labels with colors in the raw data table."""
    def highlight_sentiment(val):
        if val == 'positive': return 'color: #09ab3b; font-weight: bold;'
        elif val == 'negative': return 'color: #ff4b4b; font-weight: bold;'
        return 'color: gray;'
    return df.style.map(highlight_sentiment, subset=['sentiment_label'])

# --- 4. STREAMLIT USER INTERFACE ---
st.title("📊 ML Financial News Sentiment Analyzer")
st.markdown("""
    <div style='background-color: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 5px solid #4facfe; margin-bottom: 20px;'>
        <h4 style='margin:0; color: white;'>Uncover Market Sentiment in Seconds 🚀</h4>
        <p style='margin:0; color: #a6accd;'>This tool fetches live stock news from Finnhub, analyzes the text using the <b>FinBERT AI model</b>, and calculates a weighted mathematical sentiment score for the company.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for User Inputs
st.sidebar.header("⚙️ Configuration")
api_key_input = st.sidebar.text_input("Finnhub API Key", type="password", help="Get a free key from finnhub.io")
ticker_input = st.sidebar.text_input("Stock Ticker (e.g., AAPL, MSFT)", "AAPL").upper()

# Date Picker
today = datetime.now()
seven_days_ago = today - timedelta(days=7)
start_date = st.sidebar.date_input(" Start Date", seven_days_ago)
end_date = st.sidebar.date_input(" End Date", today)

st.sidebar.divider()
analyze_button = st.sidebar.button("🚀 Analyze Sentiment", use_container_width=True, type="primary")

# --- 5. MAIN APP LOGIC ---
if analyze_button:
    if not api_key_input:
        st.error(" Please enter your Finnhub API Key in the sidebar.")
    elif start_date > end_date:
        st.error(" Start Date cannot be after End Date.")
    else:
        # Step 1: Load Model
        with st.spinner(" Loading FinBERT AI Model..."):
            analyzer = load_model()
            
        # Step 2: Fetch Data
        raw_news = fetch_stock_news(ticker_input, start_date, end_date, api_key_input)
        
        if not raw_news:
            st.warning(f" No news found for {ticker_input} in that date range.")
        else:
            # Step 3: Clean Data
            with st.spinner(" Cleaning and organizing data..."):
                df_clean = clean_news_data(raw_news)
                
            # Step 4: Analyze Sentiment
            with st.spinner(" AI is reading and scoring the articles..."):
                df_scored = analyze_sentiment(df_clean, analyzer)
                df_overall = calculate_overall_sentiment(df_scored)
                
            # Step 5: Display Results!
            st.success(" Analysis Complete!")
            st.divider()
            
            st.subheader(f" Overall Sentiment for {ticker_input}")
            
            # Display metrics with interactive deltas
            col1, col2, col3 = st.columns(3)
            score = round(df_overall['overall_sentiment_score'].iloc[0], 2)
            majority = df_overall['majority_sentiment'].iloc[0].title()
            
            with col1:
                st.metric("Total Articles Analyzed", df_overall['total_articles'].iloc[0], delta="Relevant News")
            with col2:
                delta_color = "normal" if majority == "Positive" else "inverse" if majority == "Negative" else "off"
                st.metric("Majority Sentiment", majority, delta="AI Consensus", delta_color=delta_color)
            with col3:
                st.metric("Average Score (-1 to 1)", score, delta=f"{abs(score * 100):.0f}% Intensity", delta_color="normal" if score > 0 else "inverse" if score < 0 else "off")
                
            st.divider()
            
            # Use Tabs for better organization
            tab1, tab2 = st.tabs([" Data Visualization", " Detailed Article Breakdown"])
            
            with tab1:
                plot_sentiment(df_overall)
                st.info(" **How to read this chart:** Scores above 0 indicate positive sentiment, while scores below 0 indicate negative sentiment. The closer to 1 or -1, the stronger the AI's confidence.")
                
            with tab2:
                st.markdown("### Raw AI Scoring Data")
                # Display the styled DataFrame
                styled_df = style_dataframe(df_scored[['datetime', 'news', 'sentiment_label', 'confidence', 'numeric_score']])
                st.dataframe(styled_df, use_container_width=True, height=400)

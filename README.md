ML Financial News Sentiment Analyzer

An interactive web application built with Streamlit that leverages Machine Learning to analyze the sentiment of financial news in real-time. By fetching live market news and processing it through a state-of-the-art NLP model, this tool provides investors and analysts with a clear, mathematical measure of market sentiment for any given stock ticker.

✨ Key Features

Live News Fetching: Integrates directly with the Finnhub API to pull the latest headlines and summaries for any stock ticker (e.g., AAPL, MSFT, AMZN).

Smart API Limit Bypassing: Implements a custom day-by-day data fetching loop with built-in rate limiting (0.5s pauses) to safely bypass free-tier API restrictions and fetch deep historical data without crashing.

State-of-the-Art AI (FinBERT): Utilizes ProsusAI/finbert, a BERT-based Large Language Model fine-tuned specifically on financial texts, to accurately classify news as Positive, Negative, or Neutral.

Weighted Sentiment Scoring: Goes beyond simple labels by combining the AI's classification with its confidence level to generate a precise numeric sentiment score ranging from -1.0 (highly negative) to +1.0 (highly positive).

Interactive Dashboard: * Sleek UI with delta metrics to track total articles, majority sentiment, and average intensity.

Interactive visualizations powered by Plotly to visualize overall company sentiment.

Color-coded raw data tables for transparent, row-by-row analysis.

🛠️ Tech Stack

Frontend: Streamlit (Python-based UI framework)

Machine Learning: Hugging Face Transformers (PyTorch pipeline)

Data Manipulation: Pandas

Data Visualization: Plotly Express

APIs: Finnhub REST API

⚙️ How It Works

User Input: The user inputs a Finnhub API Key, a target Stock Ticker, and a custom date range in the sidebar.

Data Ingestion: The app safely queries the Finnhub API day-by-day, gathering all relevant news articles within the specified timeframe.

Data Cleaning: The raw JSON data is converted into a Pandas DataFrame, UNIX timestamps are translated into human-readable dates, and headlines/summaries are combined to give the AI full context.

AI Inference: The text is passed into the cached FinBERT pipeline. The model calculates the polarity and confidence, which is mathematically transformed into a numeric score.

Aggregation & Visualization: The data is grouped by company ticker to find the true mathematical average and majority sentiment, which is then rendered beautifully on the interactive dashboard.

🚀 Future Roadmap

Support for multi-ticker comparison in a single run.

Time-series line charts tracking sentiment changes over a given week or month.

Deployment to Streamlit Community Cloud.

import os
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from telegram import Bot
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from datetime import datetime

# Load environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Setup logging
logging.basicConfig(
    filename='news_predictor.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

NEWS_SITES = [
    'https://www.moneycontrol.com/news/business/markets/',
    'https://economictimes.indiatimes.com/markets',
    'https://www.business-standard.com/markets'
]

def gather_news():
    news = []
    for site in NEWS_SITES:
        try:
            resp = requests.get(site, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            headlines = [h.get_text() for h in soup.find_all('h2')][:10]
            news.extend(headlines)
            logging.info(f"Scraped {len(headlines)} headlines from {site}")
        except Exception as e:
            logging.error(f"Error scraping {site}: {e}")
    return news

def analyze_sentiment(news):
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(item)['compound'] for item in news]
    avg_score = np.mean(scores) if scores else 0
    logging.info(f"Sentiment scores: {scores}")
    return avg_score

def predict_market(sentiment_score):
    # Placeholder: simple threshold model
    if sentiment_score > 0.1:
        return 'up', sentiment_score
    elif sentiment_score < -0.1:
        return 'down', sentiment_score
    else:
        return 'neutral', sentiment_score

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            bot = Bot(token=TELEGRAM_BOT_TOKEN)
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logging.info("Telegram message sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send Telegram message: {e}")
    else:
        logging.warning("Telegram credentials not set.")

def main():
    news = gather_news()
    sentiment_score = analyze_sentiment(news)
    prediction, prob = predict_market(sentiment_score)
    msg = f"Market Prediction: {prediction}\nSentiment Score: {prob:.2f}\nTop News:\n" + '\n'.join(news[:5])
    send_telegram_message(msg)
    print(msg)
    # Log prediction and news
    logging.info(f"Prediction: {prediction}, Score: {prob:.2f}, News: {news[:5]}")

if __name__ == '__main__':
    main()

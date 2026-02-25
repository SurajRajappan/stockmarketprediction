import os
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from telegram import Bot
from sklearn.linear_model import LogisticRegression
import nltk
import feedparser
import spacy
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
import yfinance as yf
import csv
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
    # Scrape headlines from main sites
    for site in NEWS_SITES:
        try:
            resp = requests.get(site, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            headlines = [h.get_text() for h in soup.find_all('h2')][:20]
            news.extend(headlines)
            logging.info(f"Scraped {len(headlines)} headlines from {site}")
        except Exception as e:
            logging.error(f"Error scraping {site}: {e}")
    # Scrape from RSS feeds
    rss_feeds = [
        'https://www.moneycontrol.com/rss/markets.xml',
        'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
        'https://www.business-standard.com/rss/markets-1.xml'
    ]
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:
                news.append(entry.title)
            logging.info(f"Scraped {len(feed.entries[:20])} RSS entries from {feed_url}")
        except Exception as e:
            logging.error(f"Error scraping RSS {feed_url}: {e}")
    return news

def fetch_market_result():
    # Fetch Nifty, BankNifty, and Sensex closing prices
    try:
        nifty = yf.Ticker("^NSEI")
        banknifty = yf.Ticker("^NSEBANK")
        sensex = yf.Ticker("^BSESN")
        hist_nifty = nifty.history(period="1d")
        hist_banknifty = banknifty.history(period="1d")
        hist_sensex = sensex.history(period="1d")
        close_nifty = hist_nifty['Close'].iloc[-1] if not hist_nifty.empty else None
        close_banknifty = hist_banknifty['Close'].iloc[-1] if not hist_banknifty.empty else None
        close_sensex = hist_sensex['Close'].iloc[-1] if not hist_sensex.empty else None
        return {
            'nifty': close_nifty,
            'banknifty': close_banknifty,
            'sensex': close_sensex
        }
    except Exception as e:
        logging.error(f"Error fetching market result: {e}")
        return {'nifty': None, 'banknifty': None, 'sensex': None}

def analyze_sentiment(news):
    # Advanced NLP: topic extraction, NER, transformer sentiment
    nlp = spacy.load('en_core_web_sm')
    sentiment_model = pipeline('sentiment-analysis')
    topics = set()
    entities = set()
    scores = []
    for item in news:
        doc = nlp(item)
        for ent in doc.ents:
            entities.add(ent.text)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                topics.add(token.lemma_)
        try:
            result = sentiment_model(item)[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            scores.append(score)
        except Exception as e:
            logging.error(f"Transformer sentiment error: {e}")
    avg_score = np.mean(scores) if scores else 0
    logging.info(f"Sentiment scores: {scores}")
    logging.info(f"Topics: {list(topics)}")
    logging.info(f"Entities: {list(entities)}")
    return avg_score, list(topics), list(entities)

def predict_market(sentiment_score):
    # Use models for Nifty, BankNifty, Sensex if trained, else threshold
    try:
        import joblib
        X = np.array([[sentiment_score]])
        preds = {}
        probs = {}
        for idx, name in enumerate(['nifty','banknifty','sensex']):
            model_file = f"model_{name}.pkl"
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0][1]
                preds[name] = 'up' if pred == 1 else 'down'
                probs[name] = prob
            else:
                # Fallback threshold
                if sentiment_score > 0.1:
                    preds[name] = 'up'
                    probs[name] = sentiment_score
                elif sentiment_score < -0.1:
                    preds[name] = 'down'
                    probs[name] = sentiment_score
                else:
                    preds[name] = 'neutral'
                    probs[name] = sentiment_score
        return preds, probs
    except Exception as e:
        logging.error(f"Model prediction error: {e}")
        # Fallback threshold for all
        preds = {}
        probs = {}
        for name in ['nifty','banknifty','sensex']:
            if sentiment_score > 0.1:
                preds[name] = 'up'
                probs[name] = sentiment_score
            elif sentiment_score < -0.1:
                preds[name] = 'down'
                probs[name] = sentiment_score
            else:
                preds[name] = 'neutral'
                probs[name] = sentiment_score
        return preds, probs

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
    sentiment_score, topics, entities = analyze_sentiment(news)
    preds, probs = predict_market(sentiment_score)
    actual_result = fetch_market_result()
    now = datetime.now()
    # Message every 4 hours: market trend
    if now.hour % 4 == 0:
        msg = (
            f"Market Trend Update\n"
            f"Nifty: {preds['nifty']} (prob: {probs['nifty']:.2f})\n"
            f"BankNifty: {preds['banknifty']} (prob: {probs['banknifty']:.2f})\n"
            f"Sensex: {preds['sensex']} (prob: {probs['sensex']:.2f})\n"
            f"Sentiment Score: {sentiment_score:.2f}\n"
            f"Topics: {', '.join(topics[:10])}\n"
            f"Entities: {', '.join(entities[:10])}\n"
            f"Top News:\n" + '\n'.join(news[:5])
        )
        send_telegram_message(msg)
        print(msg)
    # 9 AM IST: morning probabilities
    if now.hour == 3 and now.minute == 30:  # 9 AM IST is 3:30 UTC
        msg = (
            f"Morning Market Probabilities\n"
            f"Nifty: {preds['nifty']} (prob: {probs['nifty']:.2f})\n"
            f"BankNifty: {preds['banknifty']} (prob: {probs['banknifty']:.2f})\n"
            f"Sensex: {preds['sensex']} (prob: {probs['sensex']:.2f})\n"
            f"Sentiment Score: {sentiment_score:.2f}\n"
            f"Topics: {', '.join(topics[:10])}\n"
            f"Entities: {', '.join(entities[:10])}\n"
            f"Top News:\n" + '\n'.join(news[:5])
        )
        send_telegram_message(msg)
        print(msg)
    # 3:15 PM IST: next day prediction
    if now.hour == 9 and now.minute == 45:  # 15:15 IST is 9:45 UTC
        msg = (
            f"Next Day Market Trend Prediction\n"
            f"Nifty: {preds['nifty']} (prob: {probs['nifty']:.2f})\n"
            f"BankNifty: {preds['banknifty']} (prob: {probs['banknifty']:.2f})\n"
            f"Sensex: {preds['sensex']} (prob: {probs['sensex']:.2f})\n"
            f"Sentiment Score: {sentiment_score:.2f}\n"
            f"Top News:\n" + '\n'.join(news[:5])
        )
        send_telegram_message(msg)
        print(msg)
    # 11 PM IST: analyse prediction vs actual
    if now.hour == 17 and now.minute == 30:  # 23:00 IST is 17:30 UTC
        # Compare last prediction and actual
        try:
            df = pd.read_csv('history.csv', header=None, names=['date','sentiment','prediction_nifty','nifty','banknifty','sensex'])
            last = df.iloc[-1]
            msg = (
                f"Prediction Analysis\n"
                f"Last Prediction: Nifty {last['prediction_nifty']}\n"
                f"Actual Nifty Close: {last['nifty']}\n"
                f"BankNifty Close: {last['banknifty']}\n"
                f"Sensex Close: {last['sensex']}\n"
                f"Sentiment Score: {last['sentiment']}\n"
            )
            send_telegram_message(msg)
            print(msg)
        except Exception as e:
            logging.error(f"Error in prediction analysis: {e}")
    # Log prediction, news, and actual result
    logging.info(
        f"Prediction: {preds}, Score: {probs}, Actual: {actual_result}, News: {news[:5]}"
    )
    # Save to history.csv
    with open('history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(), sentiment_score, preds['nifty'], actual_result['nifty'], actual_result['banknifty'], actual_result['sensex']
        ])
    # Retrain models if enough data
    retrain_model()

def retrain_model():
    try:
        import joblib
        if not os.path.exists('history.csv'):
            return
        df = pd.read_csv('history.csv', header=None, names=['date','sentiment','prediction_nifty','nifty','banknifty','sensex'])
        # Only train if enough data
        if len(df) < 10:
            return
        for name in ['nifty','banknifty','sensex']:
            # Convert prediction to binary (up=1, down=0)
            df_valid = df[df[f'prediction_{name}'].isin(['up','down'])]
            df_valid[f'target_{name}'] = np.where(df_valid[name].shift(-1) > df_valid[name], 1, 0)
            X = df_valid[['sentiment']].values
            y = df_valid[f'target_{name}'].values
            if len(y) < 10:
                continue
            model = LogisticRegression()
            model.fit(X, y)
            import joblib
            joblib.dump(model, f'model_{name}.pkl')
            logging.info(f"Model retrained and saved for {name}.")
    except Exception as e:
        logging.error(f"Error retraining model: {e}")
if __name__ == '__main__':
    main()

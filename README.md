# Stock News Predictor

A Python project to gather Indian stock market news, analyze sentiment, predict market direction, and send Telegram notifications. Runs on free runners/cloud VMs (e.g., GitHub Actions).

## Features
- Scrapes news from Indian financial sites
- Performs sentiment analysis
- Predicts market direction (up/down probability)
- Sends daily suggestions via Telegram bot
- Logs predictions and news sources

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set up Telegram bot token and chat ID in `.env`
3. Run main script: `python main.py`

## GitHub Actions
Automated workflow runs overnight to gather news and send predictions.

## License
Open-source, zero cost.

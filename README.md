# Stock Sentiment Analyser

An AI-powered web app that analyses the sentiment of live financial news headlines for any stock or company, built with Python and deployed on Streamlit Cloud.

## What it does

- Fetches the latest financial news headlines for any company using the NewsAPI
- Runs each headline through **FinBERT**, an AI model specifically trained on financial text
- Classifies each headline as positive, negative, or neutral
- Displays an overall sentiment score, a colour-coded headline table, and a sentiment breakdown chart

## Why I built this

I built this project to develop practical AI and Python skills relevant to finance and fintech roles. It demonstrates how natural language processing can be applied to real financial data; a technique used by quantitative analysts and fintech companies to inform trading decisions.

## Tech stack

- Python
- [FinBERT](https://huggingface.co/ProsusAI/finbert) - NLP model trained on financial text
- Streamlit - web app framework
- NewsAPI - live financial news data
- Pandas - data processing

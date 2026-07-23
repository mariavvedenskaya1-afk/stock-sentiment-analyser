# Stock Sentiment Analyser

An AI-powered web app that analyses the sentiment of live financial news headlines for any stock or company, built with Python and deployed on Streamlit Cloud.

Live at: https://stocksentimentanalyser.streamlit.app/

<img width="1390" height="535" alt="Screenshot 2026-07-16 at 15 54 37" src="https://github.com/user-attachments/assets/f4ebd4b1-5f0f-4e8f-ba92-9219c0d9ba5f" />


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

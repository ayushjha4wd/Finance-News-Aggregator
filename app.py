import requests
import json
import faiss
import numpy as np
import torch
import os
import warnings
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from database import init_db, save_news, get_all_news, clear_news

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask App
app = Flask(__name__)
CORS(app)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["100 per day", "10 per hour"])

# Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set custom cache directory
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Load Hugging Face Models
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", cache_folder=CACHE_DIR).to(device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", from_tf=True, cache_dir=CACHE_DIR)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1, cache_dir=CACHE_DIR)

# NewsAPI Key
NEWS_API_KEY = "352f67b35a544f408c58c74c654cfd7e"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# FAISS Vector Database Setup
vector_dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(vector_dim)

# Initialize SQLite DB
init_db()

# Fetch News from API
def fetch_news():
    params = {
        "q": "finance",
        "language": "en",
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        print(f"⚠️ Error Fetching News: {response.status_code}, {response.text}")
        return []

# Summarize News Articles
def summarize_text(text, max_length=100):
    try:
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"⚠️ Summarization Error: {e}")
        return text[:max_length]

# Categorize News
def categorize_news(text):
    labels = ["Stock Market", "Cryptocurrency", "Forex & Currency", "Economics & Policy", "Personal Finance"]
    try:
        result = classifier(text, candidate_labels=labels)
        return result["labels"][0]
    except Exception as e:
        print(f"⚠️ Classification Error: {e}")
        return "Uncategorized"

# Process and Store News
def process_and_store_news():
    global index
    articles = fetch_news()
    clear_news()  # Clear old data
    index.reset()

    for article in articles:
        title = article.get("title", "No Title")
        description = article.get("description", "")
        url = article.get("url", "#")
        content = f"{title}. {description}"

        summary = summarize_text(content)
        category = categorize_news(summary)
        embedding = embedding_model.encode(summary, convert_to_tensor=True).cpu().detach().numpy().reshape(1, -1)

        save_news(title, description, url, summary, category, embedding)
        index.add(embedding)

    return len(articles)

# Search News with FAISS
def search_news(query, k=5):
    news_data = get_all_news()
    if not news_data:
        return {"message": "⚠️ No news indexed. Please fetch news first."}

    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().detach().numpy().reshape(1, -1)
    D, I = index.search(query_embedding, k)

    results = []
    for i in I[0]:
        if 0 <= i < len(news_data):
            results.append({k: v for k, v in news_data[i].items() if k != "embedding"})

    return results if results else {"message": "⚠️ No relevant news found."}

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/fetch_news", methods=["GET"])
@limiter.limit("5 per minute")
def fetch_and_store():
    try:
        count = process_and_store_news()
        return jsonify({"message": f"✅ Indexed {count} news articles!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/search", methods=["GET"])
@limiter.limit("10 per minute")
def search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    try:
        return jsonify(search_news(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
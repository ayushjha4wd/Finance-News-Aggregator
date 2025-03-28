import requests
import json
import faiss
import numpy as np
import torch
import os
import warnings
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from database import init_db, save_news, get_all_news, clear_news

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
CORS(app)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["100 per day", "10 per hour"])

# Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Set custom cache directory
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Load Fast Models
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=CACHE_DIR).to(device)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", cache_dir=CACHE_DIR)
    classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base", device=0 if torch.cuda.is_available() else -1, cache_dir=CACHE_DIR)
    logger.info("Fast models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

# NewsAPI Key
NEWS_API_KEY = "352f67b35a544f408c58c74c654cfd7e"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# FAISS Vector Database Setup
vector_dim = embedding_model.get_sentence_embedding_dimension()  # Now 384 for MiniLM
index = faiss.IndexFlatL2(vector_dim)

# Initialize SQLite DB
try:
    init_db()
    logger.info("Database initialized")
except Exception as e:
    logger.error(f"Database initialization failed: {str(e)}")
    raise

# Fetch News from API
def fetch_news():
    params = {
        "q": "finance",
        "language": "en",
        "apiKey": NEWS_API_KEY
    }
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles:
            logger.warning("No articles returned from NewsAPI")
        return articles
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {str(e)}")
        raise

# Summarize News Articles
def summarize_text(text, max_length=100):
    try:
        input_length = len(text.split())
        max_length = min(max_length, input_length // 2)  # Adjust dynamically
        summary = summarizer(text, max_length=max_length, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.warning(f"Summarization error: {str(e)}. Using fallback.")
        return text[:max_length]

# Categorize News
def categorize_news(text):
    labels = ["Stock Market", "Cryptocurrency", "Forex & Currency", "Economics & Policy", "Personal Finance"]
    try:
        result = classifier(text, candidate_labels=labels)
        return result["labels"][0]
    except Exception as e:
        logger.warning(f"Classification error: {str(e)}. Using fallback.")
        return "Uncategorized"

# Process and Store News
def process_and_store_news():
    global index
    try:
        articles = fetch_news()
        if not articles:
            return 0
        clear_news()
        index.reset()
        logger.info(f"Fetched {len(articles)} articles")

        processed_count = 0
        for article in articles:
            try:
                title = article.get("title", "No Title")
                description = article.get("description", "") or ""
                url = article.get("url", "#")
                content = f"{title}. {description}"

                summary = summarize_text(content)
                category = categorize_news(summary)
                embedding = embedding_model.encode(summary, convert_to_tensor=True).cpu().detach().numpy()
                if embedding.shape != (vector_dim,):
                    embedding = embedding.reshape(1, -1)

                save_news(title, description, url, summary, category, embedding)
                index.add(embedding)
                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to process article '{title}': {str(e)}")
                continue

        logger.info(f"Processed and stored {processed_count} articles")
        return processed_count
    except Exception as e:
        logger.error(f"Error in process_and_store_news: {str(e)}")
        raise

# Search News with FAISS
def search_news(query, k=5):
    try:
        news_data = get_all_news()
        if not news_data:
            return {"message": "⚠️ No news indexed. Please fetch news first."}

        query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().detach().numpy().reshape(1, -1)
        D, I = index.search(query_embedding, k)

        results = []
        for i in I[0]:
            if 0 <= i < len(news_data):
                article = news_data[i]
                embedding = np.frombuffer(article["embedding"], dtype=np.float32)
                results.append({k: v for k, v in article.items() if k != "embedding"})

        return results if results else {"message": "⚠️ No relevant news found."}
    except Exception as e:
        logger.error(f"Error in search_news: {str(e)}")
        raise

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/fetch_news", methods=["GET"])
@limiter.limit("5 per minute")
def fetch_and_store():
    try:
        count = process_and_store_news()
        logger.info(f"Indexed {count} articles")
        return jsonify({"message": f"✅ Indexed {count} news articles!"})
    except Exception as e:
        logger.error(f"Fetch news endpoint error: {str(e)}")
        return jsonify({"error": f"Failed to fetch news: {str(e)}"}), 500

@app.route("/api/search", methods=["GET"])
@limiter.limit("10 per minute")
def search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    try:
        results = search_news(query)
        logger.info(f"Search for '{query}' returned {len(results) if isinstance(results, list) else 0} results")
        return jsonify(results)
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        return jsonify({"error": f"Failed to search: {str(e)}"}), 500

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
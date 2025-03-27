import os
import requests
import faiss
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ Load Sentence Transformer Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)  # 🔄 Updated Model

# ✅ NewsAPI Key (Secure)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "352f67b35a544f408c58c74c654cfd7e")
API_URL = f"https://newsapi.org/v2/everything?q=finance&language=en&apiKey={NEWS_API_KEY}"

# ✅ Define News Categories
NEWS_CATEGORIES = {
    "Stock Market": {"stock market", "equity", "IPO", "earnings report"},
    "Cryptocurrency": {"crypto", "bitcoin", "ethereum", "blockchain"},
    "Forex & Currency": {"forex", "currency exchange", "USD", "EUR"},
    "Economics & Policy": {"GDP", "inflation", "monetary policy", "central bank"},
    "Personal Finance": {"investment", "retirement", "tax saving", "mutual funds"},
}

# ✅ FAISS Vector Database Setup (Updated to 768D for new model)
vector_dim = 768  # 🔄 Updated Embedding Dimension
index = faiss.IndexFlatL2(vector_dim)
news_texts = []

# ✅ Fetch News from API
def fetch_news():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching news: {e}")
        return []

# ✅ Categorize and Embed News Articles
def categorize_and_store_news():
    global news_texts, index
    articles = fetch_news()
    categorized_news = []

    for article in articles:
        title = article.get("title", "No Title")
        description = article.get("description", "No Description")
        content = f"{title}. {description}"

        for category, keywords in NEWS_CATEGORIES.items():
            if any(keyword in content.lower() for keyword in keywords):
                categorized_news.append((category, content))
                break

    if categorized_news:
        embeddings = model.encode([text for _, text in categorized_news], convert_to_tensor=True).cpu().detach().numpy()
        index.add(embeddings)
        news_texts = categorized_news

    return len(categorized_news)

# ✅ Search News with FAISS
def search_news(query, k=5):
    if not news_texts:
        return {"error": "⚠️ No news indexed. Please fetch and categorize news first."}

    query_vector = model.encode([query], convert_to_tensor=True).cpu().detach().numpy()
    D, I = index.search(query_vector, k)

    results = []
    for i in I[0]:
        if i < len(news_texts):
            results.append({"category": news_texts[i][0], "news": news_texts[i][1]})

    return results if results else {"error": "⚠️ No relevant news found."}

# ✅ Flask API Endpoints
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the AI Financial News API!", "endpoints": ["/fetch_news", "/search?query=bitcoin"]})

@app.route("/fetch_news", methods=["GET"])
def fetch_and_store():
    count = categorize_and_store_news()
    return jsonify({"message": f"✅ Indexed {count} news articles!"})

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    return jsonify(search_news(query))

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

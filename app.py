import os
import requests
import json
import faiss
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# ✅ Set Hugging Face Cache to a writable directory
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface_cache"

# ✅ Ensure the directory exists
if not os.path.exists("/tmp/huggingface_cache"):
    os.makedirs("/tmp/huggingface_cache", exist_ok=True)

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ Select a Transformer Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_names = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1"
]

model = None
for model_name in model_names:
    try:
        model = SentenceTransformer(model_name, cache_folder="/tmp/huggingface_cache").to(device)
        print(f"✅ Loaded model: {model_name}")
        break
    except Exception as e:
        print(f"⚠️ Failed to load {model_name}: {str(e)}")

if model is None:
    raise RuntimeError("❌ No suitable SentenceTransformer model could be loaded.")

# ✅ Initialize FAISS Index
embedding_dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)
news_articles = []

# ✅ Set Your News API Key
NEWS_API_KEY = "352f67b35a544f408c58c74c654cfd7e"
NEWS_API_URL = "https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey=" + NEWS_API_KEY

# ✅ Fetch News from API
def fetch_news():
    response = requests.get(NEWS_API_URL)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []

# ✅ Index News Articles
def index_news():
    global news_articles, index
    news_articles = fetch_news()
    index.reset()

    if not news_articles:
        print("⚠️ No news articles fetched.")
        return
    
    # Extract headlines for embedding
    headlines = [article["title"] for article in news_articles]
    embeddings = model.encode(headlines, convert_to_numpy=True)

    # Add embeddings to FAISS index
    index.add(embeddings)
    print(f"✅ Indexed {len(headlines)} news articles.")

# ✅ Search News Articles
@app.route("/search", methods=["POST"])
def search_news():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k=5)

    results = []
    for i in I[0]:
        if i < len(news_articles):
            results.append(news_articles[i])

    return jsonify(results)

# ✅ Refresh News Index
@app.route("/refresh", methods=["GET"])
def refresh_news():
    index_news()
    return jsonify({"message": "News index updated", "total_articles": len(news_articles)})

# ✅ Start the Application
if __name__ == "__main__":
    index_news()
    app.run(host="0.0.0.0", port=5000, debug=True)

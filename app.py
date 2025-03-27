import os
import requests
import json
import faiss
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# ✅ Set Hugging Face Cache Directory
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface_cache"
if not os.path.exists("/tmp/huggingface_cache"):
    os.makedirs("/tmp/huggingface_cache")

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ Load Sentence Transformer Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name).to(device)
print(f"✅ Loaded model: {model_name}")

# ✅ FAISS Index for News Articles
d = 768  # Model embedding size
index = faiss.IndexFlatL2(d)
news_articles = []

# ✅ Fetch & Index News
def fetch_and_index_news():
    global news_articles
    API_KEY = "352f67b35a544f408c58c74c654cfd7e"
    URL = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={API_KEY}"

    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        
        news_articles.clear()
        vectors = []
        
        for article in articles:
            title = article.get("title", "")
            content = article.get("description", "")
            full_text = f"{title}. {content}"
            
            news_articles.append(full_text)
            embedding = model.encode(full_text, convert_to_numpy=True)
            vectors.append(embedding)
        
        if vectors:
            index.add(np.array(vectors))
            print(f"✅ Indexed {len(news_articles)} news articles.")
    else:
        print("❌ Failed to fetch news:", response.status_code)

fetch_and_index_news()

# ✅ Search API Endpoint
@app.route("/search", methods=["POST"])
def search_news():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
    D, I = index.search(query_embedding, 5)

    results = [{"news": news_articles[i], "score": float(D[0][j])} for j, i in enumerate(I[0]) if i < len(news_articles)]
    return jsonify({"query": query, "results": results})

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

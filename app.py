import requests
import json
import faiss
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load Hugging Face Models
embedding_model = SentenceTransformer("BAAI/bge-large-en").to(device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

# ✅ NewsAPI Key 
NEWS_API_KEY = "352f67b35a544f408c58c74c654cfd7e"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# ✅ FAISS Vector Database Setup
vector_dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(vector_dim)
news_data = []  # Stores news articles with category, summary, and embedding

# ✅ Fetch News from API
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

# ✅ Summarize News Articles
def summarize_text(text, max_length=100):
    try:
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"⚠️ Summarization Error: {e}")
        return text[:max_length]  # Return truncated text if summarization fails

# ✅ Categorize News Using Zero-Shot Classification
def categorize_news(text):
    labels = ["Stock Market", "Cryptocurrency", "Forex & Currency", "Economics & Policy", "Personal Finance"]
    try:
        result = classifier(text, candidate_labels=labels)
        return result["labels"][0]  # Return best-matching category
    except Exception as e:
        print(f"⚠️ Classification Error: {e}")
        return "Uncategorized"

# ✅ Fetch, Process, and Store News
def process_and_store_news():
    global news_data, index
    articles = fetch_news()
    news_data.clear()
    index.reset()  # Clear FAISS index before reloading

    for article in articles:
        title = article.get("title", "No Title")
        description = article.get("description", "")
        url = article.get("url", "#")
        content = f"{title}. {description}"

        # Summarize and categorize news
        summary = summarize_text(content)
        category = categorize_news(summary)

        # Encode news text into vector
        embedding = embedding_model.encode(summary, convert_to_tensor=True).cpu().detach().numpy().reshape(1, -1)

        # Store in FAISS and local list
        index.add(embedding)
        news_data.append({"category": category, "summary": summary, "url": url})

    return len(news_data)

# ✅ Search News with FAISS
def search_news(query, k=5):
    if not news_data:
        return {"message": "⚠️ No news indexed. Please fetch news first."}

    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().detach().numpy().reshape(1, -1)
    D, I = index.search(query_embedding, k)

    results = []
    for i in I[0]:
        if 0 <= i < len(news_data):
            results.append(news_data[i])

    return results if results else {"message": "⚠️ No relevant news found."}

# ✅ Flask API Endpoints
@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to the AI Financial News Aggregator API!",
        "endpoints": ["/fetch_news", "/search?query=bitcoin"]
    })

@app.route("/fetch_news", methods=["GET"])
def fetch_and_store():
    count = process_and_store_news()
    return jsonify({"message": f"✅ Indexed {count} news articles!"})

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    return jsonify(search_news(query))

# ✅ Run Flask App (Without ngrok for Local Testing)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

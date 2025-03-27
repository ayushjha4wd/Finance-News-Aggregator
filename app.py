import os
import json
import faiss
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

# ✅ Set a user-accessible cache directory to avoid permission issues
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"

# ✅ Ensure the directory exists
if not os.path.exists("/tmp/huggingface_cache"):
    os.makedirs("/tmp/huggingface_cache")

# ✅ Load the Sentence Transformer model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
try:
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Loaded model: {MODEL_NAME}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ✅ Load and index news articles
NEWS_FILE = "news.json"  # JSON file containing news articles
news_articles = []
news_embeddings = None
index = None

def load_news():
    global news_articles, news_embeddings, index
    try:
        with open(NEWS_FILE, "r") as file:
            news_articles = json.load(file)
        print(f"✅ Loaded {len(news_articles)} news articles.")

        # ✅ Convert news headlines into embeddings
        news_texts = [article["title"] for article in news_articles]
        news_embeddings = model.encode(news_texts, normalize_embeddings=True)

        # ✅ Create FAISS index
        d = news_embeddings.shape[1]  # Embedding dimension
        index = faiss.IndexFlatL2(d)
        index.add(np.array(news_embeddings, dtype=np.float32))
        print(f"✅ Indexed {len(news_texts)} news articles.")

    except Exception as e:
        print(f"❌ Error loading news data: {e}")
        exit(1)

# 🔄 Load news articles when the script starts
load_news()

# ✅ Flask app setup
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Finance News API is running!"})

@app.route("/search", methods=["GET"])
def search_news():
    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400

    try:
        # ✅ Encode the query
        query_embedding = model.encode([query], normalize_embeddings=True)

        # ✅ Perform similarity search in FAISS
        k = 5  # Number of results
        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)

        # ✅ Retrieve matching articles
        results = [{"title": news_articles[i]["title"], "url": news_articles[i]["url"]} for i in indices[0]]

        return jsonify({"query": query, "results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

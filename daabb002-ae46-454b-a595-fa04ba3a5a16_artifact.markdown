# Finance News Aggregator

A web application that fetches, summarizes, and indexes finance-related news articles, with search functionality and a chatbot for finance queries. Built with Flask, hosted on Hugging Face Spaces.

Live at: [https://ayush2917-finance-news-api.hf.space/](https://ayush2917-finance-news-api.hf.space/)

## Features
- **News Fetching**: Pulls the latest finance news from NewsAPI.
- **Summarization**: Uses `distilbart-cnn-6-6` to summarize articles.
- **Categorization**: Classifies articles into finance categories (e.g., Stock Market, Cryptocurrency) using `nli-distilroberta-base`.
- **Search**: Indexes articles with FAISS for fast similarity search.
- **Chatbot**: Answers finance questions using `gpt2-medium`, leveraging news context when available.

## Tech Stack
- **Backend**: Flask (Python)
- **AI Models**: SentenceTransformers (`all-MiniLM-L6-v2`), Transformers (`distilbart-cnn-6-6`, `nli-distilroberta-base`, `gpt2-medium`)
- **Database**: SQLite (`/app/data/news.db`)
- **Vector Search**: FAISS
- **Hosting**: Hugging Face Spaces
- **Dependencies**: Listed in `requirements.txt`

## Setup (Local Development)
For local testing or modification:

1. **Clone the Repository**:
   ```bash
   git clone https://huggingface.co/spaces/ayush2917/finance-news-api
   cd finance-news-api
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   ```bash
   python app.py
   ```
   - Access at `http://localhost:7860`.

4. **Environment**:
   - Ensure Python 3.9+.
   - NewsAPI key is hardcoded (`352f67b35a544f408c58c74c654cfd7e`); replace with your own if needed.

## Usage (Hosted on Hugging Face Spaces)
The app is live and requires no setup to use via the web interface or API.

### Web Interface
- Visit [https://ayush2917-finance-news-api.hf.space/](https://ayush2917-finance-news-api.hf.space/).
- **Fetch News**: Click the "Fetch News" button (triggers `/api/fetch_news`).
- **Search**: Enter a query (e.g., "Bitcoin") in the search bar.
- **Chat**: Type a finance question (e.g., "What’s the stock market trend today?") in the chat box.

### API Endpoints
Interact programmatically using these endpoints:

1. **Fetch News**:
   - **Endpoint**: `GET /api/fetch_news`
   - **Description**: Fetches and indexes finance news.
   - **Rate Limit**: 5/minute
   - **Example**:
     ```bash
     curl https://ayush2917-finance-news-api.hf.space/api/fetch_news
     ```
   - **Response**: `{"message": "✅ Indexed 99 news articles!"}`

2. **Search News**:
   - **Endpoint**: `GET /api/search?query=<query>`
   - **Description**: Searches indexed news for relevant articles.
   - **Rate Limit**: 10/minute
   - **Example**:
     ```bash
     curl "https://ayush2917-finance-news-api.hf.space/api/search?query=Bitcoin"
     ```
   - **Response**: JSON list of articles or `{"message": "⚠️ No news indexed..."}`

3. **Chat with Bot**:
   - **Endpoint**: `POST /api/chat`
   - **Description**: Answers finance queries, using news context if available.
   - **Rate Limit**: 20/minute
   - **Example**:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"query":"bitcoin price trends"}' https://ayush2917-finance-news-api.hf.space/api/chat
     ```
   - **Response**: `{"response": "Bitcoin prices are volatile, driven by recent crypto news."}`

## Project Structure
```
finance-news-api/
├── app.py              # Main Flask app
├── database.py         # SQLite database functions
├── Dockerfile          # Container setup for Spaces
├── requirements.txt    # Python dependencies
├── static/
│   └── style.css       # Frontend styling
├── templates/
│   └── index.html      # Frontend template
└── README.md           # This file
```

## How It Works
1. **News Fetching**: Hits NewsAPI with "finance" query, processes up to 100 articles.
2. **Processing**: Summarizes with `distilbart`, categorizes with `nli-distilroberta`, embeds with `all-MiniLM-L6-v2`.
3. **Storage**: Saves to SQLite, indexes embeddings in FAISS.
4. **Search**: Uses FAISS to find top 5 similar articles for a query.
5. **Chatbot**: `gpt2-medium` generates responses based on news or general knowledge.

## Troubleshooting
- **Chatbot Fails**: If "Sorry, I couldn’t process your query" appears, check logs for model errors; ensure news is fetched for context.
- **Search Returns 0**: Fetch news first via `/api/fetch_news`.
- **Slow Responses**: Hosted on CPU-only Spaces; expect slight delays with `gpt2-medium`.
- **Logs**: View in Spaces "Logs" tab for debugging (e.g., `INFO - Indexed 99 articles`).

## Limitations
- **NewsAPI Quota**: Free tier limits requests; upgrade for production use.
- **Chatbot**: `gpt2-medium` may give generic answers without news context; fine-tuning could improve it.
- **Hosting**: Spaces’ CPU limits performance; GPU support requires a paid plan.

## Future Improvements
- Fine-tune `gpt2-medium` on finance data.
- Add real-time stock/crypto price integration.
- Enhance UI with article previews and chat history.
- Support multiple languages via NewsAPI.

## Credits
- Built by [ayush2917](https://huggingface.co/ayush2917).
- Powered by NewsAPI, Hugging Face Transformers, and xAI’s Grok assistance.

Last Updated: March 28, 2025
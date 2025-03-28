import sqlite3
import os

def init_db():
    os.makedirs("/app/data", exist_ok=True)
    conn = sqlite3.connect("/app/data/news.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        url TEXT,
        summary TEXT,
        category TEXT,
        embedding BLOB
    )''')
    conn.commit()
    conn.close()

def save_news(title, description, url, summary, category, embedding):
    conn = sqlite3.connect("/app/data/news.db")
    c = conn.cursor()
    c.execute("INSERT INTO news (title, description, url, summary, category, embedding) VALUES (?, ?, ?, ?, ?, ?)",
              (title, description, url, summary, category, sqlite3.Binary(embedding.tobytes())))
    conn.commit()
    conn.close()

def get_all_news():
    conn = sqlite3.connect("/app/data/news.db")
    c = conn.cursor()
    c.execute("SELECT * FROM news")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "description": r[2], "url": r[3], "summary": r[4], "category": r[5], "embedding": r[6]} for r in rows]

def clear_news():
    conn = sqlite3.connect("/app/data/news.db")
    c = conn.cursor()
    c.execute("DELETE FROM news")
    conn.commit()
    conn.close()
# setup_db.py
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

# pgvector拡張を有効化（初回のみ必要）
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# ドキュメントテーブルを作成
# vector(768) は text-embedding-004 の次元数
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id          SERIAL PRIMARY KEY,
        title       TEXT NOT NULL,
        body        TEXT NOT NULL,
        category    TEXT,
        created_at  TIMESTAMP DEFAULT NOW(),
        embedding   vector(768)
    );
""")

conn.commit()
cur.close()
conn.close()

print("テーブル作成完了")
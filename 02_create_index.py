# create_index.py
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

# HNSWインデックスを作成
# vector_cosine_ops = コサイン類似度で検索する指定
cur.execute("""
    CREATE INDEX IF NOT EXISTS docs_embedding_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (
        m = 16,              -- 各ノードの最大接続数（精度とメモリのトレードオフ）
        ef_construction = 64 -- 構築時の探索幅（大きいほど精度↑・構築時間↑）
    );
""")

conn.commit()
print("インデックス作成完了")
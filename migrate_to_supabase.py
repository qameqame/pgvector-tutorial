# migrate_to_supabase.py
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# ── ローカルDBに接続 ──────────────────────────────────────────
local_conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
local_cur = local_conn.cursor()

# ── Supabaseに接続 ────────────────────────────────────────────
supa_conn = psycopg2.connect(
    host=os.getenv("SUPABASE_HOST"),
    port=os.getenv("SUPABASE_PORT"),
    dbname=os.getenv("SUPABASE_DB"),
    user=os.getenv("SUPABASE_USER"),
    password=os.getenv("SUPABASE_PASSWORD"),
    sslmode="require",  # Supabaseは SSL必須
)
supa_cur = supa_conn.cursor()

# ── ローカルのデータを取得 ────────────────────────────────────
local_cur.execute("SELECT title, body, category, embedding FROM documents;")
rows = local_cur.fetchall()
print(f"移行するドキュメント数: {len(rows)}")

# ── Supabaseに挿入 ────────────────────────────────────────────
for row in rows:
    title, body, category, embedding = row
    supa_cur.execute("""
        INSERT INTO documents (title, body, category, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
    """, (title, body, category, embedding))

supa_conn.commit()
print("移行完了！")

# 確認
supa_cur.execute("SELECT COUNT(*) FROM documents;")
count = supa_cur.fetchone()[0]
print(f"Supabase内のドキュメント数: {count}")

local_conn.close()
supa_conn.close()
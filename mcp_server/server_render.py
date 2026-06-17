# mcp_server/server_render.py
import psycopg2
from google import genai
from google.genai import types as genai_types
from fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()

mcp = FastMCP(
    name="pgvector-search",
    instructions="pgvectorを使ったドキュメント検索サーバーです。"
                 "機械学習・Python・クラウドに関するドキュメントを検索できます。"
)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── DB接続（環境変数から取得） ──────────────────────────────────
# ローカル: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
# Render:  DATABASE_URL（Renderが自動設定、またはSupabaseのURLを設定）
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Render / Supabase の接続文字列を使う
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
else:
    # ローカル用フォールバック
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_HOST", os.getenv("DB_HOST")),
        port=os.getenv("SUPABASE_PORT", os.getenv("DB_PORT")),
        dbname=os.getenv("SUPABASE_DB", os.getenv("DB_NAME")),
        user=os.getenv("SUPABASE_USER", os.getenv("DB_USER")),
        password=os.getenv("SUPABASE_PASSWORD", os.getenv("DB_PASSWORD")),
        sslmode="require",
    )

cur = conn.cursor()


def get_embedding(text: str) -> list[float]:
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


@mcp.tool
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """
    全カテゴリのドキュメントからクエリに関連するものを検索する。

    Args:
        query: 検索クエリ
        top_k: 取得するドキュメント数（デフォルト: 3）

    Returns:
        タイトル・本文・カテゴリ・類似度スコアのリスト
    """
    query_embedding = get_embedding(query)
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]


@mcp.tool
def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
    """
    特定カテゴリのドキュメントだけを検索する。

    Args:
        query: 検索クエリ
        category: カテゴリ名（ML / Python / Cloud）
        top_k: 取得するドキュメント数（デフォルト: 3）

    Returns:
        タイトル・本文・カテゴリ・類似度スコアのリスト
    """
    query_embedding = get_embedding(query)
    cur.execute("""
        SELECT title, body, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        WHERE category = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, category, query_embedding, top_k))
    rows = cur.fetchall()
    return [
        {"title": r[0], "body": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]


@mcp.tool
def list_categories() -> list[dict]:
    """
    DBに存在するカテゴリとドキュメント数の一覧を返す。

    Returns:
        カテゴリ名とドキュメント数のリスト
    """
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


@mcp.resource("db://categories")
def get_categories_resource() -> str:
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    lines = [f"- {r[0]}: {r[1]}件" for r in rows]
    return "利用可能なカテゴリ:\n" + "\n".join(lines)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # RenderはPORT環境変数を自動設定
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
    )
# 04_search.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={"api_version": "v1"})

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


def get_query_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values

def search(query: str, top_k: int = 3) -> list[dict]:
    """クエリに意味的に近いドキュメントを返す"""

    # クエリをベクトル化
    query_embedding = get_query_embedding(query)

    # <=> 演算子 = コサイン距離（0に近いほど類似）
    # 1 - コサイン距離 = コサイン類似度（1に近いほど類似）
    cur.execute("""
        SELECT
            id,
            title,
            category,
            1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))

    rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]


# 検索を実行（基本）
results = search("機械学習のモデル精度を測る方法", top_k=3)
for r in results:
    print(f"[{r['similarity']:.4f}] {r['title']} ({r['category']})")


def search_with_filter(query: str, category: str = None, top_k: int = 3) -> list[dict]:
    query_embedding = get_query_embedding(query)
    where_clause = "WHERE 1=1"
    params = [query_embedding, query_embedding]  # ← まずembeddingを2つ
    if category:
        where_clause += " AND category = %s"
        params = [query_embedding, category, query_embedding]  # ← categoryをembeddingの間に挟む
    params.append(top_k)
    cur.execute(f"""
        SELECT
            id,
            title,
            category,
            1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, params)
    rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "category": r[2], "similarity": round(r[3], 4)}
        for r in rows
    ]


# MLカテゴリのみで検索（フィルタ付き）
results = search_with_filter("Pythonでモデルを評価したい", category="ML", top_k=3)
for r in results:
    print(f"[{r['similarity']:.4f}] {r['title']} ({r['category']})")
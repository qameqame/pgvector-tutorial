# 05_rag.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

# Gemini初期化
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# DB接続
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


def search_with_filter(query: str, category: str = None, top_k: int = 3) -> list[dict]:
    query_embedding = get_query_embedding(query)
    where_clause = "WHERE 1=1"
    params = [query_embedding, query_embedding]
    if category:
        where_clause += " AND category = %s"
        params.append(category)
    params.append(top_k)
    cur.execute(f"""
        SELECT id, title, category,
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


def get_body(doc_id: int) -> str:
    cur.execute("SELECT body FROM documents WHERE id = %s", (doc_id,))
    row = cur.fetchone()
    return row[0] if row else ""


def rag_answer(question: str, category: str = None) -> str:
    """Vector DBで関連文書を取得し、LLMに回答させる"""

    # Step 1: 関連ドキュメントを検索
    docs = search_with_filter(question, category=category, top_k=3)

    if not docs:
        return "関連するドキュメントが見つかりませんでした。"

    # Step 2: 検索結果をコンテキストとして整形
    context = "\n\n".join([
        f"【{d['title']}】\n{get_body(d['id'])}"
        for d in docs
    ])

    # Step 3: LLMに渡すプロンプトを構築
    prompt = f"""以下のドキュメントを参考に、質問に答えてください。

# 参考ドキュメント
{context}

# 質問
{question}

# 回答（参考ドキュメントに基づいて簡潔に）"""

    response = client.models.generate_content(
       model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

# 実行
answer = rag_answer("F1スコアはどう計算しますか？")
print(answer)
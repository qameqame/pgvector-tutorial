# observability/traced_rag.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse import get_client, observe
langfuse = get_client()
import time

load_dotenv()

# ── Langfuseクライアントの初期化 ──────────────────────────────
langfuse = Langfuse()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


# ── @observe() デコレーターでトレースを記録 ───────────────────
# 関数の実行時間・入出力が自動的にLangfuseに送られる

@observe()  # ← これだけでトレースされる
def get_embedding(text: str) -> list[float]:
    """Embeddingの生成をトレース"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


@observe()  # ← Vector DB検索もトレース
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Vector DB検索をトレース"""
    query_embedding = get_embedding(query)
    cur.execute("""
        SELECT title, body,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    results = [
        {"title": r[0], "body": r[1], "similarity": round(r[2], 4)}
        for r in rows
    ]

    # トレースにメタデータを追加
    langfuse.update_current_observation(
        metadata={"retrieved_count": len(results), "top_similarity": results[0]["similarity"] if results else 0}
    )
    return results


@observe(name="llm_generate")  # ← 名前を指定してトレース
def generate_answer(question: str, context: str) -> str:
    """LLM回答生成をトレース"""
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


@observe(name="rag_pipeline")  # ← RAG全体をトレース
def rag_answer(question: str) -> str:
    """
    RAGパイプライン全体をトレース
    Langfuseのダッシュボードで以下が確認できる:
    - rag_pipeline（全体）
      ├── search_documents（Vector DB検索）
      │   └── get_embedding（Embedding生成）
      └── llm_generate（LLM回答生成）
    """
    # トレースにユーザーの質問を記録
    langfuse.update_current_trace(
        name="rag_pipeline",
        input=question,
        tags=["rag", "production"],
    )

    # Step 1: ドキュメント検索
    docs = search_documents(question, top_k=3)

    # Step 2: コンテキスト整形
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])

    # Step 3: LLM回答生成
    answer = generate_answer(question, context)

    # トレースに最終回答を記録
    langfuse.update_current_trace(output=answer)
    return answer


# ── 実行 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "F1スコアはどう計算しますか？",
        "AWSのコストを最適化する方法は？",
        "Pandasで欠損値を処理するには？",
    ]

    for question in questions:
        print(f"\n質問: {question}")
        answer = rag_answer(question)
        print(f"回答: {answer[:100]}...")
        time.sleep(3)  # レート制限対策

    # Langfuseにトレースを送信（非同期バッファをフラッシュ）
    langfuse.flush()
    print("\nLangfuseにトレースを送信しました")
    print("https://cloud.langfuse.com でダッシュボードを確認してください")
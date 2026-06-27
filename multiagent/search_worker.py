# multiagent/search_worker.py
"""
検索専門ワーカー

役割: pgvectorからドキュメントを検索して返す
責任: 検索のみ（回答生成はしない）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

# 検索ワーカーのシステムプロンプト
# 単一の責任に集中したシンプルなプロンプト
SEARCH_WORKER_SYSTEM = """あなたはドキュメント検索の専門家です。
与えられた質問に関連するドキュメントをpgvectorから検索し、
検索結果をそのまま返してください。

制約:
- 回答を生成しない（検索結果を返すだけ）
- 検索結果がない場合は「関連ドキュメントが見つかりませんでした」と返す
- カテゴリが明確な場合はカテゴリ絞り込みを使う
"""


def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """全カテゴリから検索"""
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


def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
    """カテゴリ絞り込み検索"""
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


def list_categories() -> list[dict]:
    """カテゴリ一覧を取得"""
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


# ── Tool定義 ─────────────────────────────────────────────────
SEARCH_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="全カテゴリのドキュメントからクエリに関連するものを検索する",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="検索クエリ"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="取得件数"),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_by_category",
            description="特定カテゴリのドキュメントだけを検索する",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING),
                    "category": types.Schema(type=types.Type.STRING, description="ML / Python / Cloud"),
                    "top_k": types.Schema(type=types.Type.INTEGER),
                },
                required=["query", "category"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_categories",
            description="DBのカテゴリ一覧を返す",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
    ]
)


def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "search_by_category":
        return search_by_category(**func_args)
    elif func_name == "list_categories":
        return list_categories()
    return {"error": f"unknown: {func_name}"}


def run_search_worker(question: str) -> dict:
    """
    検索ワーカーのメイン関数

    Args:
        question: 検索する質問

    Returns:
        {
            "docs": [検索されたドキュメントのリスト],
            "steps": [実行したステップ数],
            "worker": "search"
        }
    """
    print(f"  [検索ワーカー] 質問: {question[:40]}...")

    contents = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    docs = []
    steps = 0

    for _ in range(5):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SEARCH_WORKER_SYSTEM,
                        tools=[SEARCH_TOOLS],
                    ),
                )
                break
            except Exception as e:
                if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                    time.sleep((attempt + 1) * 10)
                else:
                    raise

        candidates = response.candidates
        if not candidates or not candidates[0].content.parts:
            break

        part = candidates[0].content.parts[0]

        if part.function_call:
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            result = dispatch(func_name, func_args)
            steps += 1
            print(f"  [検索ワーカー] {func_name} → {len(result) if isinstance(result, list) else result}件")

            # 検索結果を蓄積
            if func_name in ("search_documents", "search_by_category") and isinstance(result, list):
                docs.extend(result)

            contents.append(
                types.Content(role="model", parts=[types.Part(function_call=part.function_call)])
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response={"result": result},
                        )
                    )]
                )
            )
        else:
            break

    # 重複除去（タイトルで）
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc["title"] not in seen:
            seen.add(doc["title"])
            unique_docs.append(doc)

    print(f"  [検索ワーカー] 完了: {len(unique_docs)}件のドキュメントを取得")
    return {
        "docs": unique_docs,
        "steps": steps,
        "worker": "search",
    }
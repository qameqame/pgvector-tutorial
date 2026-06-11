# 09_agent_basic.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time
import json

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


# ══════════════════════════════════════════
# ツール①: Vector DB検索
# ══════════════════════════════════════════
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Vector DBからドキュメントを検索する"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    query_embedding = result.embeddings[0].values

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


# ══════════════════════════════════════════
# ツール②: カテゴリ一覧取得
# ══════════════════════════════════════════
def list_categories() -> list[dict]:
    """DBに存在するカテゴリとドキュメント数を返す"""
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


# ══════════════════════════════════════════
# ツール③: 類似度スコアの統計計算
# ══════════════════════════════════════════
def calculate_stats(scores: list[float]) -> dict:
    """類似度スコアの統計情報を計算する"""
    if not scores:
        return {"error": "スコアが空です"}
    return {
        "count": len(scores),
        "average": round(sum(scores) / len(scores), 4),
        "max": round(max(scores), 4),
        "min": round(min(scores), 4),
    }


# ══════════════════════════════════════════
# ツール定義
# ══════════════════════════════════════════
tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="ユーザーの質問に関連するドキュメントをVector DBから検索する。"
                        "情報を調べる必要があるときに使う。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="検索クエリ"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="取得件数。デフォルト3。"),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_categories",
            description="DBに存在するカテゴリとドキュメント数の一覧を取得する。"
                        "どんなカテゴリがあるか確認したいときに使う。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="calculate_stats",
            description="類似度スコアのリストを受け取り、平均・最大・最小などの統計情報を計算する。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "scores": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.NUMBER),
                        description="類似度スコアのリスト（0〜1の数値）",
                    ),
                },
                required=["scores"],
            ),
        ),
    ]
)


# ══════════════════════════════════════════
# ディスパッチャー（ツール名→関数の対応）
# ══════════════════════════════════════════
def dispatch(func_name: str, func_args: dict):
    """LLMから返ってきたツール名で実際の関数を呼び出す"""
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "list_categories":
        return list_categories()
    elif func_name == "calculate_stats":
        return calculate_stats(**func_args)
    return {"error": f"unknown function: {func_name}"}


# ══════════════════════════════════════════
# リトライ処理
# ══════════════════════════════════════════
def generate_with_retry(contents, tools_obj, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(tools=[tools_obj]),
            )
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_attempts - 1:
                wait = (attempt + 1) * 10
                print(f"サーバー混雑、{wait}秒待ってリトライ ({attempt+1}/{max_attempts-1})...")
                time.sleep(wait)
            else:
                raise


# ══════════════════════════════════════════
# Agentのメインループ
# ══════════════════════════════════════════
def agent(task: str, max_steps: int = 8) -> str:
    """
    複数ツールを自律的に組み合わせてタスクを達成するエージェント

    ReActパターン:
      Reasoning → Acting → Observation → Reasoning → ...
    """
    print(f"\nタスク: {task}")
    print("=" * 60)

    contents = [types.Content(role="user", parts=[types.Part(text=task)])]

    for step in range(max_steps):
        print(f"\n[Step {step + 1}] Reasoning...")

        response = generate_with_retry(contents, tools)

        candidates = response.candidates
        if not candidates or not candidates[0].content or not candidates[0].content.parts:
            return "（回答を取得できませんでした）"

        part = candidates[0].content.parts[0]

        if part.function_call:
            # Acting: ツールを実行する
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            print(f"  → Acting: {func_name}({func_args})")

            result = dispatch(func_name, func_args)

            # Observation: 結果を表示して履歴に追加
            if isinstance(result, list):
                print(f"  → Observation: {len(result)}件取得")
            else:
                print(f"  → Observation: {result}")

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
            # 目標達成 → ループ終了
            text_parts = [p.text for p in candidates[0].content.parts if hasattr(p, 'text') and p.text]
            print(f"\n[完了] {step + 1}ステップで達成")
            return "\n".join(text_parts) if text_parts else "（回答を取得できませんでした）"

    return "最大ステップ数に達しました。"


# ══════════════════════════════════════════
# 実行
# ══════════════════════════════════════════

# タスク①: カテゴリを調べてから検索する（複数ツールの連携）
print(agent(
    "まずDBにどんなカテゴリがあるか調べて、"
    "その後でPythonカテゴリのドキュメントを検索してください。"
))

# タスク②: 検索してから統計も計算する（3ツールの連携）
print(agent(
    "機械学習の評価指標についてのドキュメントを検索して、"
    "見つかったドキュメントの類似度スコアの統計情報も計算してください。"
))
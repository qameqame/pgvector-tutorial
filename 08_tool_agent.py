# 08_tool_agent.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
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

def generate_with_retry(contents, tools_obj, max_attempts=5):
    """503/429エラー時に自動リトライするgenerate_contentラッパー"""
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


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = get_query_embedding(query)
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
    query_embedding = get_query_embedding(query)
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


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="全カテゴリのドキュメントから関連情報を検索する。",
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
            description="特定カテゴリ（ML・Python・Cloud）のドキュメントだけを検索する。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="検索クエリ"),
                    "category": types.Schema(type=types.Type.STRING, description="カテゴリ名"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="取得件数"),
                },
                required=["query", "category"],
            ),
        ),
    ]
)


def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "search_by_category":
        return search_by_category(**func_args)
    return {"error": f"unknown function: {func_name}"}


def agent(question: str, max_steps: int = 5) -> str:
    """
    LLMが自律的にツールを呼び出し続け、
    最終回答を出すまでループする。
    max_steps で無限ループを防ぐ。
    """
    print(f"\n質問: {question}")
    print("=" * 50)

    # 会話履歴を蓄積していく
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")

        response = generate_with_retry(contents, tools)

        candidates = response.candidates
        if not candidates or not candidates[0].content or not candidates[0].content.parts:
            return "（回答を取得できませんでした）"

        part = candidates[0].content.parts[0]

        if part.function_call:
            # ツール呼び出しが要求された → 実行して会話履歴に追加
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            print(f"ツール呼び出し: {func_name}({func_args})")

            result = dispatch(func_name, func_args)
            print(f"結果: {len(result)}件取得")

            # モデルのツール要求を履歴に追加
            contents.append(
                types.Content(role="model", parts=[types.Part(function_call=part.function_call)])
            )
            # ツールの実行結果を履歴に追加
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
            # テキスト回答が返ってきた → ループ終了
            text_parts = [p.text for p in candidates[0].content.parts if hasattr(p, 'text') and p.text]
            print(f"\n最終回答（{step + 1}ステップで完了）:")
            return "\n".join(text_parts) if text_parts else "（回答を取得できませんでした）"

    return "最大ステップ数に達しました。"


# ── 実行 ────────────────────────────────────────────────────────
answer = agent(
    "機械学習の評価方法を教えてください。"
    "まず全体を調べてから、MLカテゴリでも詳しく調べてください。"
)
print(answer)
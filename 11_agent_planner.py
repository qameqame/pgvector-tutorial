# 11_agent_planner.py
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


def search_documents(query: str, top_k: int = 3) -> list[dict]:
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


def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
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
    cur.execute("SELECT category, COUNT(*) as count FROM documents GROUP BY category ORDER BY count DESC;")
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="全カテゴリから関連ドキュメントを検索する。",
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
            description="特定カテゴリのドキュメントだけを検索する。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="検索クエリ"),
                    "category": types.Schema(type=types.Type.STRING, description="カテゴリ名（ML/Python/Cloud）"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="取得件数"),
                },
                required=["query", "category"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_categories",
            description="DBに存在するカテゴリ一覧を取得する。",
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
    return {"error": f"unknown function: {func_name}"}


def generate_with_retry(contents, tools_obj, system_instruction=None, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            config_args = {"tools": [tools_obj]}
            if system_instruction:
                config_args["system_instruction"] = system_instruction
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(**config_args),
            )
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_attempts - 1:
                wait = (attempt + 1) * 10
                print(f"サーバー混雑、{wait}秒待ってリトライ ({attempt+1}/{max_attempts-1})...")
                time.sleep(wait)
            else:
                raise


# ══════════════════════════════════════════
# Phase 1: 計画（Plan）
# LLMにまず「何をどの順番でやるか」を考えさせる
# ══════════════════════════════════════════
def plan(task: str) -> str:
    print(f"\n[Phase 1] 計画を立てています...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""以下のタスクを達成するための実行計画を立ててください。
使えるツール: search_documents, search_by_category, list_categories

タスク: {task}

計画を箇条書きで簡潔に書いてください（3〜5ステップ）。""",
    )
    plan_text = response.candidates[0].content.parts[0].text
    print(f"計画:\n{plan_text}")
    return plan_text


# ══════════════════════════════════════════
# Phase 2: 実行（Execute）
# 計画に基づいてエージェントを実行する
# ══════════════════════════════════════════
def execute(task: str, plan_text: str, max_steps: int = 8) -> str:
    print(f"\n[Phase 2] 計画を実行しています...")

    system_instruction = f"""以下の計画に従ってタスクを実行してください。

計画:
{plan_text}

計画通りに進め、各ステップでツールを使って情報を収集してから最終回答を生成してください。"""

    contents = [types.Content(role="user", parts=[types.Part(text=task)])]

    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")

        response = generate_with_retry(contents, tools, system_instruction)

        candidates = response.candidates
        if not candidates or not candidates[0].content or not candidates[0].content.parts:
            return "（回答を取得できませんでした）"

        part = candidates[0].content.parts[0]

        if part.function_call:
            func_name = part.function_call.name
            func_args = dict(part.function_call.args)
            print(f"  → {func_name}({func_args})")

            result = dispatch(func_name, func_args)
            print(f"  → {len(result) if isinstance(result, list) else result}件")

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
            text_parts = [p.text for p in candidates[0].content.parts if hasattr(p, 'text') and p.text]
            return "\n".join(text_parts) if text_parts else "（回答を取得できませんでした）"

    return "最大ステップ数に達しました。"


# ══════════════════════════════════════════
# Phase 3: 評価（Evaluate）
# 結果がタスクを満たしているか評価する
# ══════════════════════════════════════════
def evaluate(task: str, result: str) -> str:
    print(f"\n[Phase 3] 結果を評価しています...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""以下のタスクと回答を評価してください。

タスク: {task}

回答: {result}

評価項目:
1. タスクの要求を満たしているか（Yes/No）
2. 不足している情報があれば指摘してください
3. 総合評価（1〜5点）

簡潔に回答してください。""",
    )
    evaluation = response.candidates[0].content.parts[0].text
    print(f"評価結果:\n{evaluation}")
    return evaluation


# ══════════════════════════════════════════
# Plan→Execute→Evaluate の統合実行
# ══════════════════════════════════════════
def plan_execute_evaluate(task: str):
    print(f"\n{'='*60}")
    print(f"タスク: {task}")
    print(f"{'='*60}")

    # Phase 1: 計画
    plan_text = plan(task)

    # Phase 2: 実行
    result = execute(task, plan_text)
    print(f"\n[実行結果]\n{result}")

    # Phase 3: 評価
    evaluate(task, result)


# ══════════════════════════════════════════
# 実行
# ══════════════════════════════════════════
plan_execute_evaluate(
    "機械学習とクラウド技術の両方について調査して、"
    "それぞれの主要なトピックをまとめてください。"
)
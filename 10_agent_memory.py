# 10_agent_memory.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time
import json
from datetime import datetime

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
# Long-term Memory: ファイルで永続化
# ══════════════════════════════════════════
MEMORY_FILE = "agent_memory.json"

def load_memory() -> dict:
    """保存済みの記憶を読み込む"""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"facts": [], "searches": []}

def save_memory(memory: dict):
    """記憶をファイルに保存する"""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════
# ツール定義（検索 + メモリ操作）
# ══════════════════════════════════════════
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


def remember_fact(fact: str) -> dict:
    """重要な情報を長期記憶に保存する"""
    memory = load_memory()
    memory["facts"].append({
        "fact": fact,
        "timestamp": datetime.now().isoformat()
    })
    save_memory(memory)
    return {"status": "保存しました", "fact": fact}


def recall_facts() -> list[dict]:
    """長期記憶から保存済みの情報を取り出す"""
    memory = load_memory()
    return memory["facts"]


def clear_memory() -> dict:
    """長期記憶をリセットする"""
    save_memory({"facts": [], "searches": []})
    return {"status": "メモリをクリアしました"}


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="Vector DBからドキュメントを検索する。",
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
            name="remember_fact",
            description="重要な情報や学んだことを長期記憶に保存する。"
                        "次回の会話でも使いたい情報はここに保存する。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "fact": types.Schema(type=types.Type.STRING, description="保存する情報"),
                },
                required=["fact"],
            ),
        ),
        types.FunctionDeclaration(
            name="recall_facts",
            description="長期記憶に保存されている情報を取り出す。"
                        "以前学んだことを確認したいときに使う。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="clear_memory",
            description="長期記憶をすべて削除してリセットする。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
    ]
)


def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "remember_fact":
        return remember_fact(**func_args)
    elif func_name == "recall_facts":
        return recall_facts()
    elif func_name == "clear_memory":
        return clear_memory()
    return {"error": f"unknown function: {func_name}"}


def generate_with_retry(contents, tools_obj, system_instruction=None, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            config = types.GenerateContentConfig(tools=[tools_obj])
            if system_instruction:
                config = types.GenerateContentConfig(
                    tools=[tools_obj],
                    system_instruction=system_instruction,
                )
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config,
            )
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_attempts - 1:
                wait = (attempt + 1) * 10
                print(f"サーバー混雑、{wait}秒待ってリトライ ({attempt+1}/{max_attempts-1})...")
                time.sleep(wait)
            else:
                raise


def agent(task: str, max_steps: int = 8) -> str:
    print(f"\nタスク: {task}")
    print("=" * 60)

    # システムプロンプト: エージェントの役割と行動指針
    system_instruction = """あなたは学習支援エージェントです。
- 重要な情報を学んだら remember_fact で長期記憶に保存してください
- タスク開始時に recall_facts で過去の記憶を確認してください
- 記憶を活用して、より質の高い回答を提供してください"""

    contents = [types.Content(role="user", parts=[types.Part(text=task)])]

    for step in range(8):
        print(f"\n[Step {step + 1}] Reasoning...")

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

            if isinstance(result, list):
                print(f"  → {len(result)}件")
            else:
                print(f"  → {result}")

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
            print(f"\n[完了] {step + 1}ステップで達成")
            return "\n".join(text_parts) if text_parts else "（回答を取得できませんでした）"

    return "最大ステップ数に達しました。"


# ══════════════════════════════════════════
# 実行
# ══════════════════════════════════════════

# 1回目: 学習して記憶に保存
print(agent(
    "F1スコアについて調べて、重要なポイントを長期記憶に保存してください。"
))

# 2回目: 記憶を使って回答（DBを再検索しなくても答えられるはず）
print(agent(
    "以前学んだF1スコアの情報を思い出して、簡単にまとめてください。"
))
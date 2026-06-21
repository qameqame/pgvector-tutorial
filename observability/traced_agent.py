# observability/traced_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langfuse import get_client, observe
import time

load_dotenv()
langfuse = get_client()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


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


@observe(name="tool_search_documents")
def search_documents(query: str, top_k: int = 3) -> list[dict]:
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


@observe(name="tool_list_categories")
def list_categories() -> list[dict]:
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="ドキュメントをVector DBから検索する。",
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
            name="list_categories",
            description="DBのカテゴリ一覧を取得する。",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
    ]
)


def dispatch(func_name: str, func_args: dict):
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "list_categories":
        return list_categories()
    return {"error": f"unknown function: {func_name}"}


@observe(name="agent_step")  # ← 各ステップをトレース
def agent_step(contents: list, step_num: int) -> tuple:
    """Agentの1ステップをトレース"""
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(tools=[tools]),
            )
            break
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 4:
                wait = (attempt + 1) * 10
                print(f"  リトライ {attempt+1}... {wait}秒待機")
                time.sleep(wait)
            else:
                raise

    candidates = response.candidates
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return None, None

    part = candidates[0].content.parts[0]

    if part.function_call:
        func_name = part.function_call.name
        func_args = dict(part.function_call.args)
        # ステップのメタデータを記録
        langfuse.update_current_span(
            metadata={"step": step_num, "tool": func_name, "args": func_args}
        )
        return part, "tool_call", candidates
    else:
        langfuse.update_current_span(
            metadata={"step": step_num, "type": "final_answer"}
        )
        return part, "final", candidates


@observe(name="agent_pipeline")  # ← Agent全体をトレース
def run_agent(task: str, max_steps: int = 5) -> str:
    """Agentパイプライン全体をトレース"""
    langfuse.update_current_span(
        metadata={"input": task, "tags": ["agent", "multi-step"]}
    )

    print(f"\nタスク: {task}")
    contents = [types.Content(role="user", parts=[types.Part(text=task)])]
    step_count = 0

    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")
        part, step_type, candidates = agent_step(contents, step + 1)

        if part is None:
            break

        if step_type == "tool_call":
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
            step_count += 1

        elif step_type == "final":
            text_parts = [p.text for p in candidates[0].content.parts if hasattr(p, 'text') and p.text]
            answer = "\n".join(text_parts) if text_parts else ""

            # 最終回答と使用ステップ数を記録
            langfuse.update_current_span(
                metadata={"output": answer, "total_steps": step_count + 1}
            )
            print(f"\n[完了] {step + 1}ステップで達成")
            return answer

    return "最大ステップ数に達しました。"


# ── 実行 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_agent(
        "まずカテゴリを確認して、MLカテゴリの評価指標について詳しく教えてください。"
    )
    print(f"\n最終回答:\n{result[:200]}...")

    langfuse.flush()
    print("\nLangfuseにトレースを送信しました")
    print("https://cloud.langfuse.com でダッシュボードを確認してください")
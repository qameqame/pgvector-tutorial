# 07_tool_multi.py
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


# ── 実際の関数 ────────────────────────────────────────────────
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """全カテゴリから検索"""
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
    """特定カテゴリに絞って検索"""
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


# ── ツール定義（2つ） ─────────────────────────────────────────
tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="複数カテゴリにまたがる質問や、カテゴリが不明な場合に全カテゴリから検索する。"
                        "「MLとCloudを比較」のように複数分野の質問は必ずこちらを使う。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="検索クエリ",
                    ),
                    "top_k": types.Schema(
                        type=types.Type.INTEGER,
                        description="取得件数",
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_by_category",
            description="ML・Python・Cloudのいずれか1つのカテゴリに明確に絞った質問のときだけ使う。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="検索クエリ",
                    ),
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="カテゴリ名。ML / Python / Cloud のいずれか。",
                    ),
                    "top_k": types.Schema(
                        type=types.Type.INTEGER,
                        description="取得件数",
                    ),
                },
                required=["query", "category"],
            ),
        ),
    ]
)

# ── ツール呼び出しを実行するディスパッチャー ──────────────────────
def dispatch(func_name: str, func_args: dict):
    """関数名に応じて実際の関数を呼び出す"""
    if func_name == "search_documents":
        return search_documents(**func_args)
    elif func_name == "search_by_category":
        return search_by_category(**func_args)
    return {"error": f"unknown function: {func_name}"}

def generate_with_retry(contents, tools_list, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                # model="gemini-2.0-flash",  # 2.5-flash → 2.0-flash に変更
                model="gemini-2.5-flash-lite",  # 軽量版を試す
                contents=contents,
                config=types.GenerateContentConfig(tools=tools_list),
            )
        except Exception as e:
            if "503" in str(e) and attempt < max_attempts - 1:
                wait = (attempt + 1) * 10  # 10秒、20秒、30秒と増やす
                print(f"サーバー混雑、{wait}秒待ってリトライ ({attempt+1}/{max_attempts-1})...")
                time.sleep(wait)
            else:
                raise

def ask_with_tools(question: str) -> str:
    print(f"\n質問: {question}")
    print("-" * 40)

    response = generate_with_retry(question, [tools])

    candidates = response.candidates
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return "（LLMから回答がありませんでした）"

    part = candidates[0].content.parts[0]

    if part.function_call:
        func_name = part.function_call.name
        func_args = dict(part.function_call.args)
        print(f"LLMが選んだツール: {func_name}({func_args})")

        result = dispatch(func_name, func_args)
        print(f"結果: {len(result)}件取得")

        final_response = generate_with_retry(
            contents=[
                types.Content(role="user", parts=[types.Part(text=question)]),
                types.Content(role="model", parts=[types.Part(function_call=part.function_call)]),
                types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response={"result": result},
                        )
                    )]
                ),
            ],
            tools_list=[tools],
        )

        # candidatesとpartsのNullチェックを追加
        final_candidates = final_response.candidates
        if not final_candidates or not final_candidates[0].content or not final_candidates[0].content.parts:
            return "（回答を取得できませんでした）"


        final_parts = final_candidates[0].content.parts

        # function_callが含まれている場合はさらに実行
        for p in final_parts:
            if p.function_call:
                func_name2 = p.function_call.name
                func_args2 = dict(p.function_call.args)
                print(f"追加ツール呼び出し: {func_name2}({func_args2})")
                result2 = dispatch(func_name2, func_args2)
                print(f"結果: {len(result2)}件取得")
                # 2回目の結果で最終回答を生成
                # (省略: 08_tool_agent.pyのループ処理に任せる)

        text_parts = [p.text for p in final_parts if hasattr(p, 'text') and p.text]
        if text_parts:
            return "\n".join(text_parts)
        else:
            return "（複数ステップが必要な質問です → 08_tool_agent.py で試してください）"

        # final_parts = final_candidates[0].content.parts
        # # デバッグ: partsの中身を確認
        # for i, p in enumerate(final_parts):
        #     print(f"part[{i}]: type={type(p)}, has_text={hasattr(p, 'text')}, text={getattr(p, 'text', None)}")
        
        # text_parts = [p.text for p in final_parts if hasattr(p, 'text') and p.text]
        # if text_parts:
        #     return "\n".join(text_parts)
        # else:
        #     return "（回答を取得できませんでした）"

    else:
        return part.text

# ── 実行 ────────────────────────────────────────────────────────
# 全体検索を使うはず
print(ask_with_tools("AWSのコスト最適化について教えてください"))

# カテゴリ絞り込みを使うはず
print(ask_with_tools("PythonでのデータサイエンスについてML分野から教えてください"))
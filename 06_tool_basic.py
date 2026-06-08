# 06_tool_basic.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
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


# ── 実際の関数（LLMが呼び出しを要求したら実行する） ──────────────
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
        SELECT title, body,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, top_k))

    rows = cur.fetchall()
    return [
        {"title": r[0], "body": r[1], "similarity": round(r[2], 4)}
        for r in rows
    ]


# ── ツール定義（LLMに「こんな関数があります」と伝えるための定義） ──
search_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",          # 関数名（実際の関数名と合わせる）
            description="ユーザーの質問に答えるために必ず呼び出すツール。"
                        "自分の知識で答えられる場合でも、必ずこのツールで検索してから回答すること。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="検索クエリ。ユーザーの質問をそのまま渡す。",
                    ),
                    "top_k": types.Schema(
                        type=types.Type.INTEGER,
                        description="取得するドキュメント数。デフォルトは3。",
                    ),
                },
                required=["query"],           # query は必須、top_k は任意
            ),
        )
    ]
)


# ── Tool Useのメイン処理 ────────────────────────────────────────
def ask_with_tool(question: str) -> str:
    print(f"\n質問: {question}")
    print("-" * 40)

    # Step 1: LLMに質問とツール定義を渡す
    # リトライ処理（最大3回）
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=question,
                config=types.GenerateContentConfig(
                    tools=[search_tool],
                    system_instruction="質問に答える前に必ず search_documents ツールを使って検索すること。",
                ),
            )
            break  # 成功したらループを抜ける
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                print(f"サーバー混雑、{(attempt+1)*5}秒待ってリトライ...")
                time.sleep((attempt + 1) * 5)
            else:
                raise  # 3回失敗したらエラーを投げる

    # Step 2: LLMがツール呼び出しを要求しているか確認する

    candidates = response.candidates
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return "（LLMから回答がありませんでした）"
    part = candidates[0].content.parts[0]

    if part.function_call:
        # LLMがツール呼び出しを要求した場合
        func_name = part.function_call.name
        func_args = dict(part.function_call.args)
        print(f"LLMがツールを要求: {func_name}({func_args})")

        # Step 3: 実際に関数を実行する
        if func_name == "search_documents":
            search_result = search_documents(**func_args)
            print(f"検索結果: {len(search_result)}件取得")

        # Step 4: 検索結果をLLMにフィードバックして最終回答を生成
        final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=question)]),
                types.Content(role="model", parts=[types.Part(function_call=part.function_call)]),
                types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response={"result": search_result},
                        )
                    )]
                ),
            ],
            config=types.GenerateContentConfig(tools=[search_tool]),
        )
        return final_response.text

    else:
        # LLMがツール不要と判断した場合（そのまま回答）
        print("LLMはツール不要と判断")
        return part.text


# ── 実行 ────────────────────────────────────────────────────────
# ツールが必要な質問（DBを検索するはず）
print(ask_with_tool("F1スコアの計算方法を教えてください"))

# ツールが不要な質問（LLMが直接答えるはず）
print(ask_with_tool("今日は何曜日ですか？"))
# mcp_server/server.py
import psycopg2
from google import genai
from google.genai import types as genai_types
from fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()

# ── FastMCPサーバーの初期化 ──────────────────────────────────
# Tool Useでは client = genai.Client(...) だったが
# MCPでは mcp = FastMCP(...) がサーバーの起点になる
mcp = FastMCP(
    name="pgvector-search",          # サーバーの名前（Claude Desktopに表示される）
    instructions="pgvectorを使ったドキュメント検索サーバーです。"
                 "機械学習・Python・クラウドに関するドキュメントを検索できます。"
)

# ── DB・Geminiの初期化 ────────────────────────────────────────
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()


def get_embedding(text: str) -> list[float]:
    """テキストをEmbeddingベクトルに変換する"""
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


# ══════════════════════════════════════════
# Tool定義（@mcp.toolデコレーターで登録）
# Tool Useでは types.FunctionDeclaration(...) を手書きしていたが
# MCPでは @mcp.tool デコレーターと型ヒントだけで自動生成される
# ══════════════════════════════════════════

@mcp.tool
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """
    全カテゴリのドキュメントからクエリに関連するものを検索する。
    カテゴリが不明な場合やカテゴリをまたぐ質問に使う。

    Args:
        query: 検索クエリ
        top_k: 取得するドキュメント数（デフォルト: 3）

    Returns:
        タイトル・本文・カテゴリ・類似度スコアのリスト
    """
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
        {
            "title": r[0],
            "body": r[1],
            "category": r[2],
            "similarity": round(r[3], 4)
        }
        for r in rows
    ]


@mcp.tool
def search_by_category(query: str, category: str, top_k: int = 3) -> list[dict]:
    """
    特定カテゴリのドキュメントだけを検索する。
    ML・Python・Cloudなど明確にカテゴリが指定された質問に使う。

    Args:
        query: 検索クエリ
        category: カテゴリ名（ML / Python / Cloud）
        top_k: 取得するドキュメント数（デフォルト: 3）

    Returns:
        タイトル・本文・カテゴリ・類似度スコアのリスト
    """
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
        {
            "title": r[0],
            "body": r[1],
            "category": r[2],
            "similarity": round(r[3], 4)
        }
        for r in rows
    ]


@mcp.tool
def list_categories() -> list[dict]:
    """
    DBに存在するカテゴリとドキュメント数の一覧を返す。
    どんなカテゴリがあるか確認したいときに使う。

    Returns:
        カテゴリ名とドキュメント数のリスト
    """
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


# ══════════════════════════════════════════
# Resource定義（@mcp.resourceデコレーターで登録）
# Resourceはツールではなく「LLMが読めるデータ」
# ドキュメントの一覧など静的な情報をResourceとして公開する
# ══════════════════════════════════════════

@mcp.resource("db://categories")
def get_categories_resource() -> str:
    """DBのカテゴリ一覧をResource（読み取り専用データ）として公開する"""
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC;
    """)
    rows = cur.fetchall()
    lines = [f"- {r[0]}: {r[1]}件" for r in rows]
    return "利用可能なカテゴリ:\n" + "\n".join(lines)


# ══════════════════════════════════════════
# Prompt定義（@mcp.promptデコレーターで登録）
# 再利用可能なプロンプトテンプレートを定義する
# ユーザーが毎回書かなくていい定型プロンプトを登録しておく
# ══════════════════════════════════════════

@mcp.prompt
def search_prompt(topic: str) -> str:
    """指定したトピックの検索プロンプトを生成する"""
    return f"""以下のトピックについてドキュメントを検索し、わかりやすくまとめてください。

トピック: {topic}

手順:
1. まず list_categories でカテゴリを確認する
2. 関連するカテゴリがあれば search_by_category で絞り込み検索
3. なければ search_documents で全体検索
4. 検索結果をもとに回答を生成する"""


# ── サーバーの起動 ────────────────────────────────────────────
if __name__ == "__main__":
    # stdio モード: Claude DesktopやMCPクライアントから起動される標準的な方法
    mcp.run()
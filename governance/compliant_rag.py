# governance/compliant_rag.py
"""
ガバナンス対応RAG

EU AI Act Article 50（透明性義務）に対応:
- ユーザーにAIと対話していることを開示
- 監査ログを記録
- リスク評価を事前に実施
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from governance.audit_logger import AuditLogger, EventType
from governance.ai_registry import get_system
from governance.risk_assessor import assess_risk

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
logger = AuditLogger()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

# EU AI Act Article 50 — チャットボット透明性開示
AI_DISCLOSURE = """
⚠️ このシステムはAI（人工知能）が生成した回答を提供します。
   回答はpgvectorデータベースの検索結果に基づいていますが、
   誤りを含む可能性があります。重要な判断には専門家にご相談ください。
"""

SYSTEM_ID = "rag-search-001"


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


def search(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = get_embedding(query)
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


def compliant_rag_answer(
    question: str,
    user_id: str = "anonymous",
    show_disclosure: bool = True,
) -> dict:
    """
    ガバナンス対応RAGパイプライン

    EU AI Act対応:
    - Article 50: AIであることを開示
    - Article 12: 監査ログを記録
    - リスク評価を実施済みのシステムのみ実行

    Returns:
        {
            "answer": 回答,
            "disclosure": AI開示文,
            "sources": 参照ドキュメント,
            "audit_event_id": 監査ログID
        }
    """
    start_time = time.time()

    # システム登録確認
    system = get_system(SYSTEM_ID)
    if not system:
        return {"error": "システムが台帳に登録されていません"}

    # クエリを監査ログに記録
    query_event = logger.log(
        event_type=EventType.QUERY,
        system_id=SYSTEM_ID,
        user_id=user_id,
        input_summary=question,
        metadata={"system_name": system.name},
    )

    # 検索
    docs = search(question, top_k=3)
    search_duration = (time.time() - start_time) * 1000

    logger.log(
        event_type=EventType.SEARCH,
        system_id=SYSTEM_ID,
        user_id=user_id,
        input_summary=question,
        output_summary=f"{len(docs)}件のドキュメントを取得",
        metadata={"docs_count": len(docs)},
        duration_ms=search_duration,
    )

    if not docs:
        return {
            "answer": "関連するドキュメントが見つかりませんでした。",
            "disclosure": AI_DISCLOSURE,
            "sources": [],
            "audit_event_id": query_event.event_id,
        }

    # 回答生成
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])
    prompt = f"""以下のドキュメントを参考に、質問に答えてください。

# 参照ドキュメント
{context}

# 質問
{question}

# 回答（ドキュメントに基づいて簡潔に）"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            answer = response.text
            break
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise

    gen_duration = (time.time() - start_time) * 1000

    logger.log(
        event_type=EventType.GENERATION,
        system_id=SYSTEM_ID,
        user_id=user_id,
        input_summary=question,
        output_summary=answer[:100],
        metadata={"answer_length": len(answer)},
        duration_ms=gen_duration,
    )

    return {
        "answer": answer,
        "disclosure": AI_DISCLOSURE if show_disclosure else "",
        "sources": [{"title": d["title"], "similarity": d["similarity"]} for d in docs],
        "audit_event_id": query_event.event_id,
    }


if __name__ == "__main__":
    print("=== ガバナンス対応RAG ===")
    print(AI_DISCLOSURE)

    result = compliant_rag_answer(
        question="F1スコアの計算方法を教えてください",
        user_id="user_001",
    )

    print(f"回答:\n{result['answer']}")
    print(f"\n参照ドキュメント:")
    for source in result["sources"]:
        print(f"  - {source['title']} (similarity: {source['similarity']})")
    print(f"\n監査ログID: {result['audit_event_id']}")
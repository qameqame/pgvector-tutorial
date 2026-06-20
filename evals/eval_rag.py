# evals/eval_rag.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
from evals.dataset import EVAL_DATASET

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


def search(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = get_query_embedding(query)
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


def rag_answer(question: str) -> tuple[str, list[dict]]:
    """RAGで回答を生成し、使用したドキュメントも返す"""
    docs = search(question, top_k=3)
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])
    prompt = f"""以下のドキュメントを参考に、質問に答えてください。

# 参考ドキュメント
{context}

# 質問
{question}

# 回答（参考ドキュメントに基づいて簡潔に）"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text, docs
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise


# ══════════════════════════════════════════
# 評価関数
# ══════════════════════════════════════════

def eval_context_recall(retrieved_docs: list[dict], expected_docs: list[str]) -> float:
    """
    Context Recall: 期待するドキュメントが検索結果に含まれているか

    スコア = 期待するドキュメントのうち実際に取得できた割合
    """
    retrieved_titles = [d["title"] for d in retrieved_docs]
    hit = sum(1 for expected in expected_docs if expected in retrieved_titles)
    return hit / len(expected_docs) if expected_docs else 0.0


def eval_answer_relevancy(answer: str, keywords: list[str]) -> float:
    """
    Answer Relevancy: 回答に期待するキーワードが含まれているか

    スコア = 期待するキーワードのうち回答に含まれた割合
    """
    hit = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return hit / len(keywords) if keywords else 0.0


def eval_faithfulness(answer: str, retrieved_docs: list[dict]) -> float:
    """
    Faithfulness: 回答が検索結果に基づいているか
    LLMを使って採点する（LLM-as-a-Judge パターン）

    スコア = 0.0〜1.0（LLMが採点）
    """
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in retrieved_docs])
    prompt = f"""以下のコンテキストと回答を評価してください。

# コンテキスト（検索で取得したドキュメント）
{context}

# 回答
{answer}

評価基準:
- 回答がコンテキストの内容に基づいているか
- コンテキストにない情報を勝手に追加していないか（ハルシネーション）

0.0〜1.0のスコアだけを返してください。説明不要。数値のみ。"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            score_text = response.text.strip()
            return float(score_text)
        except (ValueError, Exception) as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                return 0.5  # 評価失敗時はデフォルト値


def run_eval():
    """全評価データセットに対してRAGを評価する"""
    results = []

    print("RAG評価を開始します...")
    print("=" * 60)

    for item in EVAL_DATASET:
        print(f"\n[{item['id']}] {item['question']}")

        # RAGで回答を生成
        answer, retrieved_docs = rag_answer(item["question"])
        time.sleep(2)  # レート制限対策

        # 各指標を評価
        context_recall   = eval_context_recall(retrieved_docs, item["expected_docs"])
        answer_relevancy = eval_answer_relevancy(answer, item["expected_answer_keywords"])
        faithfulness     = eval_faithfulness(answer, retrieved_docs)
        time.sleep(2)

        # 総合スコア（3指標の平均）
        overall = (context_recall + answer_relevancy + faithfulness) / 3

        result = {
            "id":               item["id"],
            "question":         item["question"][:30] + "...",
            "context_recall":   round(context_recall, 2),
            "answer_relevancy": round(answer_relevancy, 2),
            "faithfulness":     round(faithfulness, 2),
            "overall":          round(overall, 2),
        }
        results.append(result)

        print(f"  Context Recall:   {context_recall:.2f}")
        print(f"  Answer Relevancy: {answer_relevancy:.2f}")
        print(f"  Faithfulness:     {faithfulness:.2f}")
        print(f"  Overall:          {overall:.2f}")

    return results


if __name__ == "__main__":
    results = run_eval()

    print("\n" + "=" * 60)
    print("評価結果サマリー")
    print("=" * 60)

    avg_recall    = sum(r["context_recall"]   for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_faith     = sum(r["faithfulness"]     for r in results) / len(results)
    avg_overall   = sum(r["overall"]          for r in results) / len(results)

    print(f"Context Recall:   {avg_recall:.2f}")
    print(f"Answer Relevancy: {avg_relevancy:.2f}")
    print(f"Faithfulness:     {avg_faith:.2f}")
    print(f"Overall:          {avg_overall:.2f}")
# llmops/ci_eval.py
"""
CI/CD用の評価スクリプト

GitHubにpushするたびに実行され、
品質基準を下回った場合はexit code 1を返してCIを失敗させます。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import json
from llmops.prompt_registry import get_prompt
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

# ── 品質基準（これを下回ったらCIが失敗する） ─────────────────
QUALITY_THRESHOLDS = {
    "context_recall": 0.80,    # 検索精度80%以上
    "answer_relevancy": 0.70,  # 回答関連性70%以上
    "overall": 0.75,           # 総合スコア75%以上
}

# 評価するプロンプトのバージョン
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "latest")


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


def rag_answer_with_prompt(question: str, prompt_version: str) -> tuple[str, list[dict]]:
    """指定バージョンのプロンプトでRAGを実行"""
    docs = search(question, top_k=3)
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])

    # プロンプトレジストリからテンプレートを取得
    prompt_template = get_prompt("rag_answer", prompt_version)
    prompt = prompt_template.format(context=context, question=question)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text, docs
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 15)
            else:
                raise


def eval_context_recall(retrieved_docs: list[dict], expected_docs: list[str]) -> float:
    retrieved_titles = [d["title"] for d in retrieved_docs]
    hit = sum(1 for expected in expected_docs if expected in retrieved_titles)
    return hit / len(expected_docs) if expected_docs else 0.0


def eval_answer_relevancy(answer: str, keywords: list[str]) -> float:
    hit = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return hit / len(keywords) if keywords else 0.0


def run_ci_eval(prompt_version: str = "latest") -> dict:
    """CI用の評価を実行してレポートを返す"""
    print(f"CI評価開始: プロンプトバージョン={prompt_version}")
    print("=" * 60)

    results = []

    for item in EVAL_DATASET:
        print(f"\n[{item['id']}] {item['question']}")

        try:
            answer, retrieved_docs = rag_answer_with_prompt(item["question"], prompt_version)
            time.sleep(3)

            context_recall = eval_context_recall(retrieved_docs, item["expected_docs"])
            answer_relevancy = eval_answer_relevancy(answer, item["expected_answer_keywords"])
            overall = (context_recall + answer_relevancy) / 2

            results.append({
                "id": item["id"],
                "context_recall": context_recall,
                "answer_relevancy": answer_relevancy,
                "overall": overall,
            })

            status = "✓" if overall >= QUALITY_THRESHOLDS["overall"] else "✗"
            print(f"  {status} Context Recall: {context_recall:.2f} | Relevancy: {answer_relevancy:.2f} | Overall: {overall:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": item["id"],
                "context_recall": 0.0,
                "answer_relevancy": 0.0,
                "overall": 0.0,
                "error": str(e),
            })

    # 集計
    avg_recall = sum(r["context_recall"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_overall = sum(r["overall"] for r in results) / len(results)

    report = {
        "prompt_version": prompt_version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": {
            "context_recall": round(avg_recall, 3),
            "answer_relevancy": round(avg_relevancy, 3),
            "overall": round(avg_overall, 3),
        },
        "thresholds": QUALITY_THRESHOLDS,
        "passed": (
            avg_recall >= QUALITY_THRESHOLDS["context_recall"] and
            avg_relevancy >= QUALITY_THRESHOLDS["answer_relevancy"] and
            avg_overall >= QUALITY_THRESHOLDS["overall"]
        ),
        "results": results,
    }

    return report


if __name__ == "__main__":
    report = run_ci_eval(PROMPT_VERSION)

    print("\n" + "=" * 60)
    print("CI評価レポート")
    print("=" * 60)
    print(f"プロンプトバージョン: {report['prompt_version']}")
    print(f"Context Recall:   {report['metrics']['context_recall']:.3f} (基準: {QUALITY_THRESHOLDS['context_recall']})")
    print(f"Answer Relevancy: {report['metrics']['answer_relevancy']:.3f} (基準: {QUALITY_THRESHOLDS['answer_relevancy']})")
    print(f"Overall:          {report['metrics']['overall']:.3f} (基準: {QUALITY_THRESHOLDS['overall']})")

    # レポートをJSONで保存（GitHub Actionsのアーティファクトとして保存可能）
    with open("llmops/eval_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nレポートを llmops/eval_report.json に保存しました")

    if report["passed"]:
        print("\n✅ CI評価: PASSED — デプロイ可能")
        sys.exit(0)
    else:
        print("\n❌ CI評価: FAILED — 品質基準を満たしていません")
        sys.exit(1)  # CIを失敗させる
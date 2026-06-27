# multiagent/orchestrator.py
"""
オーケストレーター

役割: ユーザーの質問を受け取り、ワーカーに指示して結果を統合する
責任: タスク分解・ワーカー呼び出し・結果統合
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types
from dotenv import load_dotenv
from multiagent.search_worker import run_search_worker
from multiagent.quality_worker import run_quality_worker
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# オーケストレーターのシステムプロンプト
ORCHESTRATOR_SYSTEM = """あなたはタスクを分解して複数のワーカーに指示するオーケストレーターです。

あなたの役割:
1. ユーザーの質問を受け取る
2. 検索ワーカーが取得したドキュメントをもとに回答を生成する
3. 品質チェックの結果を確認して必要なら回答を改善する
4. 最終回答をユーザーに返す

制約:
- 参照ドキュメントに基づいて回答すること
- 品質スコアが0.7未満の場合は回答を改善すること
- 簡潔で正確な回答を心がけること
"""


def generate_answer(question: str, docs: list[dict]) -> str:
    """検索結果をもとに回答を生成する"""
    time.sleep(15)  # ← この行を追加（レート制限対策）
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
                config=types.GenerateContentConfig(
                    system_instruction=ORCHESTRATOR_SYSTEM,
                ),
            )
            return response.text
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise
    return "回答を生成できませんでした。"


def improve_answer(question: str, docs: list[dict], answer: str, feedback: str) -> str:
    """品質チェックのフィードバックをもとに回答を改善する"""
    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])

    prompt = f"""以下の回答を改善してください。

# 質問
{question}

# 参照ドキュメント
{context}

# 現在の回答
{answer}

# 改善フィードバック
{feedback}

# 改善した回答"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise
    return answer


def run_orchestrator(question: str) -> dict:
    """
    オーケストレーターのメイン関数

    処理フロー:
    1. 検索ワーカーに検索を依頼
    2. 検索結果をもとに回答を生成
    3. 品質チェックワーカーに評価を依頼
    4. スコアが低ければ回答を改善
    5. 最終回答を返す

    Args:
        question: ユーザーの質問

    Returns:
        {
            "answer": 最終回答,
            "quality": 品質スコア,
            "docs_count": 検索したドキュメント数,
            "improved": 改善したかどうか
        }
    """
    print(f"\n{'='*60}")
    print(f"オーケストレーター起動")
    print(f"質問: {question}")
    print(f"{'='*60}")

    # ── Step 1: 検索ワーカーに検索を依頼 ────────────────────
    print("\n[Step 1] 検索ワーカーに依頼...")
    search_result = run_search_worker(question)
    docs = search_result["docs"]

    if not docs:
        return {
            "answer": "関連するドキュメントが見つかりませんでした。",
            "quality": {"overall": 0.0},
            "docs_count": 0,
            "improved": False,
        }

    print(f"  → {len(docs)}件のドキュメントを取得")

    # ── Step 2: 回答を生成 ───────────────────────────────────
    print("\n[Step 2] 回答を生成中...")
    answer = generate_answer(question, docs)
    print(f"  → 回答生成完了（{len(answer)}文字）")

    # ── Step 3: 品質チェックワーカーに評価を依頼 ────────────
    print("\n[Step 3] 品質チェックワーカーに依頼...")
    quality = run_quality_worker(question, docs, answer)

    improved = False

    # ── Step 4: 品質が低ければ改善 ──────────────────────────
    if quality.get("overall", 0) < 0.7:
        print(f"\n[Step 4] 品質スコア {quality.get('overall')} < 0.7 → 回答を改善...")
        feedback = quality.get("feedback", "回答を改善してください")
        answer = improve_answer(question, docs, answer, feedback)
        improved = True
        print(f"  → 回答改善完了")
    else:
        print(f"\n[Step 4] 品質スコア {quality.get('overall')} ≥ 0.7 → 改善不要")

    print(f"\n{'='*60}")
    print("オーケストレーター完了")
    print(f"{'='*60}")

    return {
        "answer": answer,
        "quality": quality,
        "docs_count": len(docs),
        "improved": improved,
    }
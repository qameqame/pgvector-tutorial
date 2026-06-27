# multiagent/quality_worker.py
"""
品質チェック専門ワーカー

役割: 生成された回答の品質を評価する
責任: 品質評価のみ（検索・回答生成はしない）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 品質チェックワーカーのシステムプロンプト
QUALITY_WORKER_SYSTEM = """あなたは回答品質の評価専門家です。
与えられた「質問・参照ドキュメント・回答」のセットを評価してください。

評価基準:
1. Faithfulness（忠実性）: 回答がドキュメントに基づいているか（0.0〜1.0）
2. Relevancy（関連性）: 質問に正しく答えているか（0.0〜1.0）
3. Completeness（完全性）: 必要な情報が含まれているか（0.0〜1.0）

必ず以下のJSON形式で返してください:
{
  "faithfulness": 0.0〜1.0,
  "relevancy": 0.0〜1.0,
  "completeness": 0.0〜1.0,
  "overall": 0.0〜1.0,
  "feedback": "改善点があれば記述"
}
"""


def run_quality_worker(question: str, docs: list[dict], answer: str) -> dict:
    """
    品質チェックワーカーのメイン関数

    Args:
        question: 元の質問
        docs: 検索されたドキュメント
        answer: 評価する回答

    Returns:
        {
            "faithfulness": float,
            "relevancy": float,
            "completeness": float,
            "overall": float,
            "feedback": str,
            "worker": "quality"
        }
    """
    print(f"  [品質ワーカー] 回答品質を評価中...")

    context = "\n\n".join([f"【{d['title']}】\n{d['body']}" for d in docs])

    prompt = f"""以下を評価してください。

# 質問
{question}

# 参照ドキュメント
{context}

# 評価する回答
{answer}

JSON形式で評価結果を返してください。"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=QUALITY_WORKER_SYSTEM,
                ),
            )
            break
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep((attempt + 1) * 10)
            else:
                raise

    import json
    import re
    raw = response.text.strip()
    # JSONブロックを抽出
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            result["worker"] = "quality"
            print(f"  [品質ワーカー] Overall: {result.get('overall', 'N/A')}")
            return result
        except json.JSONDecodeError:
            pass

    # パース失敗時のデフォルト
    print(f"  [品質ワーカー] 評価パース失敗、デフォルト値を使用")
    return {
        "faithfulness": 0.5,
        "relevancy": 0.5,
        "completeness": 0.5,
        "overall": 0.5,
        "feedback": "評価を取得できませんでした",
        "worker": "quality",
    }
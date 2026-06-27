# 14_multiagent.py
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiagent.orchestrator import run_orchestrator

def main():
    questions = [
        "MLカテゴリの評価指標について詳しく教えてください",
        "AWSのコスト削減方法は？",
    ]

    for question in questions:
        result = run_orchestrator(question)

        time.sleep(30)  # ← 質問間に30秒待機
        print(f"\n最終回答:")
        print(result["answer"])
        print(f"\n品質スコア:")
        q = result["quality"]
        print(f"  Faithfulness:  {q.get('faithfulness', 'N/A')}")
        print(f"  Relevancy:     {q.get('relevancy', 'N/A')}")
        print(f"  Completeness:  {q.get('completeness', 'N/A')}")
        print(f"  Overall:       {q.get('overall', 'N/A')}")
        print(f"  改善済み:      {'Yes' if result['improved'] else 'No'}")
        print(f"  使用ドキュメント: {result['docs_count']}件")
        print("\n" + "="*60)


if __name__ == "__main__":
    main()
# finetuning/prepare_dataset.py
"""
Fine-tuning用データセットの準備

pgvectorに格納したドキュメントをもとに、
質問→回答のペアを作成します。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

# ── 手動で作成した質問・回答ペア ──────────────────────────────
# Fine-tuningには高品質なデータが最重要
# 実務では100〜1000件程度を手動で作成する
TRAINING_DATA = [
    {
        "instruction": "F1スコアとは何ですか？簡潔に説明してください。",
        "input": "",
        "output": "F1スコアはPrecision（適合率）とRecall（再現率）の調和平均です。"
                  "F1 = 2 × Precision × Recall ÷ (Precision + Recall) で計算します。"
                  "クラス不均衡がある場合に特に有効な評価指標です。",
    },
    {
        "instruction": "scikit-learnでクロスバリデーションを実行するコードを教えてください。",
        "input": "",
        "output": "from sklearn.model_selection import cross_val_score\n"
                  "from sklearn.ensemble import RandomForestClassifier\n\n"
                  "model = RandomForestClassifier()\n"
                  "scores = cross_val_score(model, X, y, cv=5, scoring='f1')\n"
                  "print(f'F1スコア: {scores.mean():.3f} ± {scores.std():.3f}')",
    },
    {
        "instruction": "Pandasで欠損値を処理する主な方法を教えてください。",
        "input": "",
        "output": "主な方法は3つあります。\n"
                  "1. 削除: df.dropna() で欠損値を含む行を削除\n"
                  "2. 補完: df.fillna(値) で特定の値に置換（平均値・中央値など）\n"
                  "3. 補間: df.interpolate() で前後の値から補間\n"
                  "数値列は平均値補完、カテゴリ列は最頻値補完が一般的です。",
    },
    {
        "instruction": "AWSのEC2コストを削減する方法を3つ挙げてください。",
        "input": "",
        "output": "1. スポットインスタンスの活用: オンデマンドより最大90%安く利用できます。"
                  "中断可能なバッチ処理に最適です。\n"
                  "2. リザーブドインスタンス: 1〜3年の契約で最大75%割引になります。\n"
                  "3. 不要リソースの削除: 未使用のEBSボリューム・EIP・スナップショットを定期的に削除します。",
    },
    {
        "instruction": "KubernetesのPodとは何ですか？",
        "input": "",
        "output": "PodはKubernetesの最小デプロイ単位です。\n"
                  "1つ以上のコンテナをまとめたグループで、同じノード上で動作し、\n"
                  "ストレージ・ネットワーク・実行仕様を共有します。\n"
                  "YAMLのマニフェストファイルで定義します。",
    },
    # データを増やす（実務では100件以上推奨）
    {
        "instruction": "機械学習のPrecisionとRecallの違いを教えてください。",
        "input": "",
        "output": "Precision（適合率）: 陽性と予測したもののうち実際に陽性だった割合。"
                  "TP / (TP + FP) で計算します。誤検知を減らしたい場合に重視します。\n"
                  "Recall（再現率）: 実際の陽性のうち正しく検出できた割合。"
                  "TP / (TP + FN) で計算します。見逃しを減らしたい場合に重視します。",
    },
    {
        "instruction": "PandasのDataFrameで特定の列を選択する方法は？",
        "input": "",
        "output": "列の選択方法は複数あります。\n"
                  "単一列: df['列名'] または df.列名\n"
                  "複数列: df[['列名1', '列名2']]\n"
                  "条件付き: df.loc[条件, '列名'] または df.iloc[行番号, 列番号]",
    },
    {
        "instruction": "AWSのS3とEBSの違いを教えてください。",
        "input": "",
        "output": "S3（Simple Storage Service）: オブジェクトストレージ。"
                  "ファイルをキーで管理。スケーラブルで安価。静的ファイル・バックアップ向け。\n"
                  "EBS（Elastic Block Store）: ブロックストレージ。"
                  "EC2インスタンスにアタッチして使うHDD/SSD。"
                  "データベース・OSディスクなどに使用。",
    },
]


def save_dataset(data: list[dict], output_file: str):
    """データセットをJSONL形式で保存する"""
    # 学習用（80%）と検証用（20%）に分割
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # JSONLファイルとして保存
    train_file = output_file.replace(".jsonl", "_train.jsonl")
    val_file = output_file.replace(".jsonl", "_val.jsonl")

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"学習データ: {len(train_data)}件 → {train_file}")
    print(f"検証データ: {len(val_data)}件 → {val_file}")
    return train_file, val_file


def format_prompt(item: dict) -> str:
    """Alpacaフォーマットに変換する"""
    if item["input"]:
        return (
            f"### Instruction:\n{item['instruction']}\n\n"
            f"### Input:\n{item['input']}\n\n"
            f"### Response:\n{item['output']}"
        )
    return (
        f"### Instruction:\n{item['instruction']}\n\n"
        f"### Response:\n{item['output']}"
    )


if __name__ == "__main__":
    os.makedirs("finetuning", exist_ok=True)

    print(f"データセット件数: {len(TRAINING_DATA)}")
    print("\n--- サンプル ---")
    print(format_prompt(TRAINING_DATA[0]))

    train_file, val_file = save_dataset(
        TRAINING_DATA,
        "finetuning/dataset.jsonl"
    )
    print("\nデータセット準備完了")
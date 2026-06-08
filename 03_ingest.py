# 03_ingest.py
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()


conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={"api_version": "v1"})

def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values

def insert_document(title: str, body: str, category: str) -> int:
    """ドキュメントをEmbeddingと一緒に格納する"""
    # タイトルと本文を結合してEmbedding生成
    text_to_embed = f"{title}\n\n{body}"
    embedding = get_embedding(text_to_embed)

    cur.execute("""
        INSERT INTO documents (title, body, category, embedding)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
    """, (title, body, category, embedding))

    doc_id = cur.fetchone()[0]
    conn.commit()
    return doc_id


# サンプルドキュメントを格納
sample_docs = [
    {
        "title": "機械学習モデルの評価指標",
        "body": "精度、再現率、F1スコアなどの評価指標について解説します。"
                "分類問題では混同行列を使って各指標を計算します。"
                "F1スコアはPrecision（適合率）とRecall（再現率）の調和平均で、"
                "F1 = 2 × Precision × Recall ÷ (Precision + Recall) で計算します。",
        "category": "ML",
    },
    {
        "title": "scikit-learnによるモデル評価",
        "body": "Pythonのscikit-learnライブラリを使ってモデルを評価する方法。"
                "cross_val_scoreやclassification_reportの使い方を説明します。",
        "category": "ML",
    },
    {
        "title": "Pandasによるデータ前処理",
        "body": "欠損値処理、型変換、外れ値の扱い方を説明します。"
                "DataFrameの基本操作とクリーニング手順を解説します。",
        "category": "Python",
    },
    {
        "title": "AWSコスト最適化の実践",
        "body": "EC2インスタンスタイプの選定、スポットインスタンス活用、"
                "不要リソースの削除によるコスト削減手法を紹介します。",
        "category": "Cloud",
    },
    {
        "title": "Kubernetes Podの基本",
        "body": "Kubernetesにおける最小デプロイ単位であるPodの概念と、"
                "YAMLによるマニフェスト定義の書き方を解説します。",
        "category": "Cloud",
    },
]

for doc in sample_docs:
    doc_id = insert_document(doc["title"], doc["body"], doc["category"])
    print(f"格納完了: id={doc_id} / {doc['title']}")
# evals/dataset.py

# 評価データセット
# 各エントリは「質問・期待する回答の要素・期待する検索ドキュメント」で構成
EVAL_DATASET = [
    {
        "id": "eval_001",
        "question": "F1スコアはどう計算しますか？",
        "expected_answer_keywords": ["Precision", "Recall", "調和平均", "2"],
        "expected_docs": ["機械学習モデルの評価指標"],
        "category": "ML",
    },
    {
        "id": "eval_002",
        "question": "scikit-learnでモデルを評価する方法は？",
        "expected_answer_keywords": ["cross_val_score", "classification_report", "scikit-learn"],
        "expected_docs": ["scikit-learnによるモデル評価"],
        "category": "ML",
    },
    {
        "id": "eval_003",
        "question": "AWSのコストを下げる方法は？",
        "expected_answer_keywords": ["EC2", "スポットインスタンス", "コスト"],
        "expected_docs": ["AWSコスト最適化の実践"],
        "category": "Cloud",
    },
    {
        "id": "eval_004",
        "question": "Pandasで欠損値を処理するには？",
        "expected_answer_keywords": ["欠損値", "DataFrame", "Pandas"],
        "expected_docs": ["Pandasによるデータ前処理"],
        "category": "Python",
    },
    {
        "id": "eval_005",
        "question": "Kubernetesのマニフェストファイルの書き方は？",
        "expected_answer_keywords": ["YAML", "Pod", "Kubernetes"],
        "expected_docs": ["Kubernetes Podの基本"],
        "category": "Cloud",
    },
    {
        "id": "eval_006",
        "question": "モデルの精度と再現率のバランスを測る指標は？",  # ← F1スコアと直接書いていない
        "expected_answer_keywords": ["F1", "Precision", "Recall"],
        "expected_docs": ["機械学習モデルの評価指標"],
        "category": "ML",
    },
    {
        "id": "eval_007",
        "question": "Pythonで機械学習モデルのクロスバリデーションをするには？",
        "expected_answer_keywords": ["cross_val_score", "scikit-learn"],
        "expected_docs": ["scikit-learnによるモデル評価"],
        "category": "ML",
    },
]
# governance/risk_assessor.py
"""
リスク評価モジュール

AIシステムのリスクを定量的に評価します。
EU AI Actのリスクベースアプローチに対応。
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RiskAssessment:
    system_id: str
    assessed_at: str
    scores: dict[str, float]   # 各リスク項目のスコア（0.0〜1.0）
    overall_risk: float        # 総合リスクスコア
    risk_level: str            # low / medium / high / critical
    mitigations: list[str]     # 推奨する対策
    next_review: str           # 次回レビュー日


# ── リスク評価チェックリスト ─────────────────────────────────
RISK_CRITERIA = {
    "data_privacy": {
        "name": "データプライバシー",
        "description": "個人情報・機密情報を扱うか",
        "weight": 0.25,
    },
    "decision_impact": {
        "name": "意思決定への影響",
        "description": "人間の重要な意思決定に影響するか",
        "weight": 0.25,
    },
    "autonomy": {
        "name": "自律性",
        "description": "人間の介在なく自律的に動くか",
        "weight": 0.20,
    },
    "bias_risk": {
        "name": "バイアスリスク",
        "description": "差別・偏見が生じるリスクがあるか",
        "weight": 0.15,
    },
    "explainability": {
        "name": "説明可能性",
        "description": "なぜそう回答したか説明できるか（低いほどリスク高）",
        "weight": 0.15,
    },
}


def assess_risk(
    system_id: str,
    data_privacy: float = 0.1,       # 0.0=なし 1.0=高リスク
    decision_impact: float = 0.2,
    autonomy: float = 0.3,
    bias_risk: float = 0.1,
    explainability: float = 0.7,     # 0.0=説明不可 1.0=完全説明可能
) -> RiskAssessment:
    """
    リスク評価を実行する

    Args:
        system_id: 評価するシステムのID
        data_privacy: データプライバシーリスク（0.0〜1.0）
        decision_impact: 意思決定への影響度（0.0〜1.0）
        autonomy: 自律性（0.0〜1.0）
        bias_risk: バイアスリスク（0.0〜1.0）
        explainability: 説明可能性（高いほど安全）

    Returns:
        RiskAssessment
    """
    scores = {
        "data_privacy": data_privacy,
        "decision_impact": decision_impact,
        "autonomy": autonomy,
        "bias_risk": bias_risk,
        "explainability": 1.0 - explainability,  # 説明可能性は逆転（高いほど安全）
    }

    # 重み付き平均でリスクスコアを計算
    overall_risk = sum(
        scores[key] * RISK_CRITERIA[key]["weight"]
        for key in scores
    )

    # リスクレベルの判定
    if overall_risk < 0.2:
        risk_level = "low"
    elif overall_risk < 0.4:
        risk_level = "medium"
    elif overall_risk < 0.7:
        risk_level = "high"
    else:
        risk_level = "critical"

    # 推奨対策の生成
    mitigations = []
    if scores["data_privacy"] > 0.5:
        mitigations.append("個人情報のマスキング・匿名化を実装する")
    if scores["decision_impact"] > 0.5:
        mitigations.append("重要な意思決定には必ず人間のレビューを挟む")
    if scores["autonomy"] > 0.5:
        mitigations.append("自律実行の範囲を制限し、人間の承認ステップを追加する")
    if scores["bias_risk"] > 0.3:
        mitigations.append("学習データのバイアス検査・多様性確保を実施する")
    if scores["explainability"] > 0.5:
        mitigations.append("回答の根拠（参照ドキュメント）を必ず提示する")
    if not mitigations:
        mitigations.append("現在のリスクレベルは許容範囲内です。定期レビューを継続してください。")

    # 次回レビュー日（リスクレベルに応じて設定）
    from datetime import timedelta
    days = {"low": 180, "medium": 90, "high": 30, "critical": 7}
    next_review = (datetime.now() + timedelta(days=days[risk_level])).strftime("%Y-%m-%d")

    return RiskAssessment(
        system_id=system_id,
        assessed_at=datetime.now().isoformat(),
        scores=scores,
        overall_risk=round(overall_risk, 3),
        risk_level=risk_level,
        mitigations=mitigations,
        next_review=next_review,
    )


if __name__ == "__main__":
    print("=== RAGシステムのリスク評価 ===\n")

    assessment = assess_risk(
        system_id="rag-search-001",
        data_privacy=0.1,       # 個人情報は扱わない
        decision_impact=0.2,    # 参考情報提供のみ・最終判断は人間
        autonomy=0.3,           # 一部自律的だが人間が確認
        bias_risk=0.1,          # 技術ドキュメントのみ・バイアス低
        explainability=0.8,     # 参照ドキュメントを提示できる
    )

    print(f"システムID: {assessment.system_id}")
    print(f"総合リスクスコア: {assessment.overall_risk}")
    print(f"リスクレベル: {assessment.risk_level.upper()}")
    print(f"\n各項目スコア:")
    for key, score in assessment.scores.items():
        name = RISK_CRITERIA[key]["name"]
        print(f"  {name}: {score:.2f}")
    print(f"\n推奨対策:")
    for mitigation in assessment.mitigations:
        print(f"  - {mitigation}")
    print(f"\n次回レビュー: {assessment.next_review}")
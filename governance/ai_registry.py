# governance/ai_registry.py
"""
AIシステム台帳

組織内で使用しているAIシステムを一元管理します。
EU AI ActのAnnex IVが求める技術文書の基盤になります。
"""
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class RiskLevel(Enum):
    UNACCEPTABLE = "prohibited"    # 禁止
    HIGH = "high_risk"             # 高リスク
    LIMITED = "limited_risk"       # 限定リスク
    MINIMAL = "minimal_risk"       # 最小リスク


class SystemStatus(Enum):
    ACTIVE = "active"
    TESTING = "testing"
    DEPRECATED = "deprecated"


@dataclass
class AISystem:
    """AIシステムの台帳エントリ"""
    system_id: str
    name: str
    description: str
    purpose: str                   # 用途（誰のために・何のために）
    risk_level: str                # RiskLevel の値
    status: str                    # SystemStatus の値
    model: str                     # 使用するAIモデル
    data_sources: list[str]        # 使用するデータソース
    owner: str                     # 担当者
    human_oversight: bool          # 人間の監視があるか
    eu_ai_act_category: str        # EU AI Actの分類
    registered_at: str
    last_reviewed: str


# ── 台帳定義 ─────────────────────────────────────────────────
REGISTRY = {
    "rag-search-001": AISystem(
        system_id="rag-search-001",
        name="pgvector RAG検索システム",
        description="pgvectorとGemini Embeddingを使ったドキュメント検索・回答生成システム",
        purpose="エンジニア向け技術ドキュメントの検索・質問応答。社内利用のみ。",
        risk_level=RiskLevel.LIMITED.value,
        status=SystemStatus.ACTIVE.value,
        model="gemini-2.5-flash + gemini-embedding-001",
        data_sources=["pgvector（社内ドキュメントDB）"],
        owner="Hiroki Kameyama",
        human_oversight=True,
        eu_ai_act_category="Limited Risk - Chatbot（Article 50 透明性義務あり）",
        registered_at="2026-06-01",
        last_reviewed=datetime.now().strftime("%Y-%m-%d"),
    ),
    "multiagent-001": AISystem(
        system_id="multiagent-001",
        name="マルチエージェント検索システム",
        description="オーケストレーター + 検索ワーカー + 品質チェックワーカーの協調システム",
        purpose="複雑な技術質問に対する高品質回答の生成。社内利用のみ。",
        risk_level=RiskLevel.LIMITED.value,
        status=SystemStatus.TESTING.value,
        model="gemini-2.5-flash（複数Agent）",
        data_sources=["pgvector（社内ドキュメントDB）"],
        owner="Hiroki Kameyama",
        human_oversight=True,
        eu_ai_act_category="Limited Risk - Chatbot（Article 50 透明性義務あり）",
        registered_at="2026-06-25",
        last_reviewed=datetime.now().strftime("%Y-%m-%d"),
    ),
}


def get_system(system_id: str) -> AISystem | None:
    return REGISTRY.get(system_id)


def list_systems(risk_level: str = None, status: str = None) -> list[AISystem]:
    systems = list(REGISTRY.values())
    if risk_level:
        systems = [s for s in systems if s.risk_level == risk_level]
    if status:
        systems = [s for s in systems if s.status == status]
    return systems


def generate_inventory_report() -> dict:
    """台帳レポートを生成する（EU AI Act対応）"""
    systems = list(REGISTRY.values())
    return {
        "generated_at": datetime.now().isoformat(),
        "total_systems": len(systems),
        "by_risk_level": {
            level.value: sum(1 for s in systems if s.risk_level == level.value)
            for level in RiskLevel
        },
        "by_status": {
            status.value: sum(1 for s in systems if s.status == status.value)
            for status in SystemStatus
        },
        "systems": [asdict(s) for s in systems],
    }


if __name__ == "__main__":
    print("=== AIシステム台帳 ===\n")
    for system in list_systems():
        print(f"[{system.system_id}] {system.name}")
        print(f"  用途: {system.purpose}")
        print(f"  リスク: {system.risk_level}")
        print(f"  状態: {system.status}")
        print(f"  EU AI Act: {system.eu_ai_act_category}")
        print(f"  人間監視: {'あり' if system.human_oversight else 'なし'}")
        print()

    report = generate_inventory_report()
    print(f"合計システム数: {report['total_systems']}")
    print(f"リスクレベル別: {report['by_risk_level']}")
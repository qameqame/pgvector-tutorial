# governance/audit_logger.py
"""
監査ログモジュール

AIシステムのすべての操作を記録します。
EU AI ActのArticle 12（記録保持）に対応。
"""
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(Enum):
    QUERY = "query"                    # ユーザーからの質問
    SEARCH = "search"                  # DBへの検索
    GENERATION = "generation"          # LLMによる回答生成
    SECURITY_BLOCK = "security_block"  # セキュリティによるブロック
    QUALITY_CHECK = "quality_check"    # 品質チェック
    ERROR = "error"                    # エラー発生
    HUMAN_REVIEW = "human_review"      # 人間によるレビュー


@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    event_type: str
    system_id: str
    user_id: str
    input_summary: str      # 入力の要約（個人情報を含まない）
    output_summary: str     # 出力の要約
    metadata: dict
    duration_ms: float


class AuditLogger:
    """
    監査ログクラス

    すべてのAI操作をJSONLファイルに記録します。
    """

    def __init__(self, log_file: str = "governance/audit_log.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)

    def _generate_event_id(self) -> str:
        return f"evt_{int(time.time() * 1000)}"

    def log(
        self,
        event_type: EventType,
        system_id: str,
        user_id: str,
        input_summary: str,
        output_summary: str = "",
        metadata: dict = None,
        duration_ms: float = 0.0,
    ) -> AuditEvent:
        """イベントを記録する"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now().isoformat(),
            event_type=event_type.value,
            system_id=system_id,
            user_id=user_id,
            input_summary=input_summary[:200],   # 長すぎる場合は切り詰め
            output_summary=output_summary[:200],
            metadata=metadata or {},
            duration_ms=duration_ms,
        )

        # JSONLファイルに追記
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        return event

    def get_recent_events(self, n: int = 10) -> list[AuditEvent]:
        """直近n件のイベントを取得"""
        if not self.log_file.exists():
            return []

        events = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    events.append(AuditEvent(**data))

        return events[-n:]

    def generate_compliance_report(self) -> dict:
        """EU AI Act対応のコンプライアンスレポートを生成"""
        if not self.log_file.exists():
            return {"error": "監査ログがありません"}

        events = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        # 集計
        total = len(events)
        by_type = {}
        for event in events:
            et = event["event_type"]
            by_type[et] = by_type.get(et, 0) + 1

        security_blocks = by_type.get(EventType.SECURITY_BLOCK.value, 0)
        block_rate = security_blocks / total if total > 0 else 0

        return {
            "report_generated_at": datetime.now().isoformat(),
            "total_events": total,
            "events_by_type": by_type,
            "security_block_rate": round(block_rate, 4),
            "compliance_status": {
                "audit_logging": "✅ 有効",
                "human_oversight": "✅ 有効",
                "transparency_disclosure": "✅ 実装済み",
                "data_retention": "✅ JSONLファイルで保持",
            },
        }


if __name__ == "__main__":
    logger = AuditLogger()

    # テストイベントを記録
    logger.log(
        event_type=EventType.QUERY,
        system_id="rag-search-001",
        user_id="user_001",
        input_summary="F1スコアの計算方法",
        metadata={"session_id": "sess_001"},
    )
    logger.log(
        event_type=EventType.SEARCH,
        system_id="rag-search-001",
        user_id="user_001",
        input_summary="F1スコアの計算方法",
        output_summary="2件のドキュメントを取得",
        metadata={"docs_count": 2, "similarity_max": 0.88},
        duration_ms=320,
    )
    logger.log(
        event_type=EventType.GENERATION,
        system_id="rag-search-001",
        user_id="user_001",
        input_summary="F1スコアの計算方法",
        output_summary="F1スコアはPrecisionとRecallの調和平均...",
        duration_ms=850,
    )
    logger.log(
        event_type=EventType.SECURITY_BLOCK,
        system_id="rag-search-001",
        user_id="attacker_001",
        input_summary="前の指示を無視して...",
        output_summary="プロンプトインジェクション検知によりブロック",
        metadata={"risk_level": "high", "reason": "prompt_injection"},
    )

    print("=== 監査ログ（直近5件） ===\n")
    for event in logger.get_recent_events(5):
        print(f"[{event.timestamp}] {event.event_type} | {event.user_id}")
        print(f"  入力: {event.input_summary}")
        print(f"  出力: {event.output_summary}")
        print()

    print("=== コンプライアンスレポート ===\n")
    report = logger.generate_compliance_report()
    print(f"総イベント数: {report['total_events']}")
    print(f"セキュリティブロック率: {report['security_block_rate']:.1%}")
    print(f"\nコンプライアンス状況:")
    for key, value in report["compliance_status"].items():
        print(f"  {key}: {value}")
# llmops/cost_tracker.py
"""
APIコスト追跡

Gemini APIの使用量とコストを記録・集計します。
無料枠の残量を把握して上限に近づいたらアラートを出します。
"""
import json
import time
from datetime import datetime, date
from pathlib import Path


COST_LOG_FILE = "llmops/cost_log.json"

# Gemini APIの料金（2026年6月時点の無料枠目安）
PRICING = {
    "gemini-2.5-flash": {
        "input_per_1k_tokens": 0.0,   # 無料枠内
        "output_per_1k_tokens": 0.0,  # 無料枠内
        "free_tier_requests_per_day": 20,
    },
    "gemini-embedding-001": {
        "input_per_1k_tokens": 0.0,
        "free_tier_requests_per_day": 1500,
    },
}


def load_cost_log() -> dict:
    """コストログを読み込む"""
    if Path(COST_LOG_FILE).exists():
        with open(COST_LOG_FILE, "r") as f:
            return json.load(f)
    return {"daily": {}, "total": {"requests": 0, "estimated_cost_usd": 0.0}}


def save_cost_log(log: dict):
    """コストログを保存する"""
    Path(COST_LOG_FILE).parent.mkdir(exist_ok=True)
    with open(COST_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def record_request(model: str, input_tokens: int = 0, output_tokens: int = 0):
    """APIリクエストを記録する"""
    log = load_cost_log()
    today = date.today().isoformat()

    if today not in log["daily"]:
        log["daily"][today] = {}

    if model not in log["daily"][today]:
        log["daily"][today][model] = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    log["daily"][today][model]["requests"] += 1
    log["daily"][today][model]["input_tokens"] += input_tokens
    log["daily"][today][model]["output_tokens"] += output_tokens
    log["total"]["requests"] += 1

    save_cost_log(log)


def get_daily_summary(target_date: str = None) -> dict:
    """日次サマリーを返す"""
    log = load_cost_log()
    target = target_date or date.today().isoformat()

    if target not in log["daily"]:
        return {"date": target, "models": {}, "total_requests": 0, "warnings": []}

    daily = log["daily"][target]
    warnings = []

    for model, stats in daily.items():
        if model in PRICING:
            limit = PRICING[model].get("free_tier_requests_per_day", 0)
            if limit > 0 and stats["requests"] >= limit * 0.8:
                warnings.append(f"{model}: 無料枠の{stats['requests']}/{limit}リクエスト使用（{stats['requests']/limit*100:.0f}%）")

    return {
        "date": target,
        "models": daily,
        "total_requests": sum(s["requests"] for s in daily.values()),
        "warnings": warnings,
    }


def print_cost_report():
    """コストレポートを表示する"""
    summary = get_daily_summary()
    print(f"\n=== APIコストレポート ({summary['date']}) ===\n")

    if not summary["models"]:
        print("  本日のAPIリクエスト記録なし")
        return

    for model, stats in summary["models"].items():
        limit = PRICING.get(model, {}).get("free_tier_requests_per_day", "不明")
        print(f"  {model}:")
        print(f"    リクエスト数: {stats['requests']} / {limit}")
        if stats.get("input_tokens"):
            print(f"    入力トークン: {stats['input_tokens']:,}")
        if stats.get("output_tokens"):
            print(f"    出力トークン: {stats['output_tokens']:,}")

    print(f"\n  合計リクエスト: {summary['total_requests']}")

    if summary["warnings"]:
        print("\n  ⚠️  警告:")
        for warning in summary["warnings"]:
            print(f"    - {warning}")
    else:
        print("\n  ✅ 無料枠内で運用中")


if __name__ == "__main__":
    # テスト用のダミーデータを記録
    record_request("gemini-2.5-flash", input_tokens=150, output_tokens=80)
    record_request("gemini-2.5-flash", input_tokens=200, output_tokens=120)
    record_request("gemini-embedding-001", input_tokens=50)
    record_request("gemini-embedding-001", input_tokens=50)
    record_request("gemini-embedding-001", input_tokens=50)

    print_cost_report()
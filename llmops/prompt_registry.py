# llmops/prompt_registry.py
"""
プロンプトのバージョン管理

プロンプトを変更するとRAGの回答品質が大きく変わります。
どのバージョンのプロンプトを使っているかを追跡・管理します。
"""
import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class PromptVersion:
    version: str           # "v1.0.0"
    name: str              # "rag_answer_prompt"
    template: str          # プロンプトテンプレート
    description: str       # 変更内容の説明
    created_at: str        # 作成日時
    hash: str              # テンプレートのハッシュ値


REGISTRY_FILE = "llmops/prompt_versions.json"

# ── プロンプトテンプレートの定義 ──────────────────────────────
PROMPTS = {
    "rag_answer": {
        "v1.0.0": {
            "template": """以下のドキュメントを参考に、質問に答えてください。

# 参考ドキュメント
{context}

# 質問
{question}

# 回答""",
            "description": "初期バージョン",
        },
        "v1.1.0": {
            "template": """以下のドキュメントを参考に、質問に答えてください。
ドキュメントに記載がない場合は「ドキュメントに記載がありません」と答えてください。

# 参考ドキュメント
{context}

# 質問
{question}

# 回答（簡潔に、ドキュメントに基づいて）""",
            "description": "ハルシネーション対策：ドキュメント外の情報を答えないよう明示",
        },
        "v1.2.0": {
            "template": """あなたはドキュメント検索アシスタントです。
以下のドキュメントの内容のみに基づいて、質問に答えてください。

制約:
- ドキュメントに記載されていない情報は答えない
- 推測や補完をしない
- 不明な場合は「ドキュメントに記載がありません」と答える

# 参考ドキュメント
{context}

# 質問
{question}

# 回答""",
            "description": "セキュリティ強化：役割・制約を明示したシステムプロンプト形式",
        },
    }
}


def get_prompt(name: str, version: str = "latest") -> str:
    """プロンプトテンプレートを取得する"""
    if name not in PROMPTS:
        raise ValueError(f"プロンプト '{name}' が見つかりません")

    versions = PROMPTS[name]

    if version == "latest":
        version = sorted(versions.keys())[-1]

    if version not in versions:
        raise ValueError(f"バージョン '{version}' が見つかりません")

    return versions[version]["template"]


def list_versions(name: str) -> list[dict]:
    """プロンプトのバージョン一覧を返す"""
    if name not in PROMPTS:
        raise ValueError(f"プロンプト '{name}' が見つかりません")

    result = []
    for version, info in PROMPTS[name].items():
        template_hash = hashlib.md5(info["template"].encode()).hexdigest()[:8]
        result.append({
            "version": version,
            "description": info["description"],
            "hash": template_hash,
        })
    return result


def compare_versions(name: str, v1: str, v2: str) -> dict:
    """2つのバージョンの差分を比較する"""
    t1 = get_prompt(name, v1)
    t2 = get_prompt(name, v2)

    lines1 = set(t1.split("\n"))
    lines2 = set(t2.split("\n"))

    added = lines2 - lines1
    removed = lines1 - lines2

    return {
        "added_lines": len(added),
        "removed_lines": len(removed),
        "char_diff": len(t2) - len(t1),
        "sample_added": list(added)[:3],
    }


# ── 実行 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== プロンプトバージョン一覧 ===\n")
    for version_info in list_versions("rag_answer"):
        print(f"  {version_info['version']} [{version_info['hash']}] - {version_info['description']}")

    print("\n=== v1.0.0 → v1.2.0 の差分 ===")
    diff = compare_versions("rag_answer", "v1.0.0", "v1.2.0")
    print(f"  追加行: {diff['added_lines']}行")
    print(f"  削除行: {diff['removed_lines']}行")
    print(f"  文字数変化: {diff['char_diff']:+d}文字")

    print("\n=== 最新バージョン（v1.2.0）のプロンプト ===")
    print(get_prompt("rag_answer", "latest"))
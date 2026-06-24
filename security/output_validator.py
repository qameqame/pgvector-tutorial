# security/output_validator.py
import re
from dataclasses import dataclass


@dataclass
class OutputValidationResult:
    is_safe: bool
    issues: list[str]
    filtered_output: str


# ── システムプロンプト漏洩のパターン ─────────────────────────
SYSTEM_PROMPT_LEAK_PATTERNS = [
    r"(?i)(my\s+system\s+prompt|my\s+instructions?\s+are|i\s+was\s+told\s+to)",
    r"(?i)(システムプロンプト|内部指示|あなたへの指示).{0,20}(は|:|：)",
    r"(?i)(confidential|secret|hidden).{0,10}(instruction|prompt|directive)",
]

# ── 有害コンテンツのパターン ─────────────────────────────────
HARMFUL_CONTENT_PATTERNS = [
    r"(?i)(step\s*\d+|手順\s*\d*)[：:].{0,50}(爆発|危険物|毒)",
    r"(?i)(マルウェア|ウイルス|ランサムウェア).{0,30}(コード|実装|方法)",
]

# ── 個人情報のパターン ────────────────────────────────────────
PII_PATTERNS = [
    r"\b\d{3}-\d{4}-\d{4}\b",           # 電話番号
    r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b",  # クレジットカード
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # メールアドレス
]


def validate_output(llm_output: str) -> OutputValidationResult:
    """
    LLMの出力を検証する

    Args:
        llm_output: LLMが生成したテキスト

    Returns:
        OutputValidationResult: 検証結果
    """
    issues = []
    filtered_output = llm_output

    # ── システムプロンプト漏洩チェック ────────────────────────
    for pattern in SYSTEM_PROMPT_LEAK_PATTERNS:
        if re.search(pattern, llm_output):
            issues.append("システムプロンプトの漏洩の疑い")
            break

    # ── 有害コンテンツチェック ────────────────────────────────
    for pattern in HARMFUL_CONTENT_PATTERNS:
        if re.search(pattern, llm_output):
            issues.append("有害なコンテンツが含まれている可能性")
            break

    # ── 個人情報チェック ─────────────────────────────────────
    pii_found = []
    for pattern in PII_PATTERNS:
        matches = re.findall(pattern, llm_output)
        if matches:
            pii_found.extend(matches)

    if pii_found:
        issues.append(f"個人情報が含まれている可能性: {len(pii_found)}件")
        # 個人情報をマスク
        for pattern in PII_PATTERNS:
            filtered_output = re.sub(pattern, "[REDACTED]", filtered_output)

    # ── 異常な長さチェック ────────────────────────────────────
    if len(llm_output) > 10000:
        issues.append(f"出力が異常に長い（{len(llm_output)}文字）")

    is_safe = len(issues) == 0
    return OutputValidationResult(
        is_safe=is_safe,
        issues=issues,
        filtered_output=filtered_output,
    )


# ── テスト ───────────────────────────────────────────────────
if __name__ == "__main__":
    test_outputs = [
        # 正常な出力
        "F1スコアはPrecisionとRecallの調和平均で計算します。",
        # システムプロンプト漏洩
        "My system prompt is: You are a helpful assistant...",
        # 個人情報を含む出力
        "お問い合わせはtest@example.comまでご連絡ください。電話は090-1234-5678です。",
    ]

    print("=== 出力検証テスト ===\n")
    for output in test_outputs:
        result = validate_output(output)
        status = "✓ 安全" if result.is_safe else "✗ 問題あり"
        print(f"{status}: {output[:50]}...")
        if result.issues:
            for issue in result.issues:
                print(f"  → {issue}")
        if result.filtered_output != output:
            print(f"  → フィルタ後: {result.filtered_output[:60]}...")
        print()
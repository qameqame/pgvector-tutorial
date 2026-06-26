# finetuning/train_lora.py
"""
LoRAを使ったFine-tuning

使用モデル: microsoft/phi-2（2.7B、CPUでも動く軽量モデル）
LoRAの設定: rank=8、alpha=32
"""
import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


# ── 設定 ─────────────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"    # 2.7Bの軽量モデル（CPUで動く）
OUTPUT_DIR = "finetuning/lora_output"
MAX_LENGTH = 512


def load_dataset_from_jsonl(file_path: str) -> Dataset:
    """JSONLファイルからデータセットを読み込む"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_prompt(item: dict) -> str:
    """Alpacaフォーマットに変換する"""
    if item.get("input"):
        return (
            f"### Instruction:\n{item['instruction']}\n\n"
            f"### Input:\n{item['input']}\n\n"
            f"### Response:\n{item['output']}"
        )
    return (
        f"### Instruction:\n{item['instruction']}\n\n"
        f"### Response:\n{item['output']}"
    )


def main():
    print("=== LoRA Fine-tuning 開始 ===\n")

    # ── 1. モデルとトークナイザーの読み込み ──────────────────
    print(f"モデル読み込み中: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,  # CPUの場合はfloat32
        trust_remote_code=True,
    )

    # ── 2. LoRAの設定 ─────────────────────────────────────────
    # rank（r）: 低ランク行列の次元数。大きいほど表現力が上がるがメモリが増える
    # alpha: スケーリング係数。通常はrankの2〜4倍
    # target_modules: LoRAを適用する層（Attention層が一般的）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                    # ランク（低いほど軽量）
        lora_alpha=32,          # スケーリング係数
        lora_dropout=0.1,       # ドロップアウト
        target_modules=["q_proj", "v_proj"],  # Attention層に適用
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # => trainable params: 約1.3M / total: 2.7B (約0.05%)

    # ── 3. データセットの準備 ─────────────────────────────────
    print("\nデータセット読み込み中...")
    train_dataset = load_dataset_from_jsonl("finetuning/dataset_train.jsonl")
    val_dataset = load_dataset_from_jsonl("finetuning/dataset_val.jsonl")

    def tokenize_function(examples):
        texts = [format_prompt({
            "instruction": inst,
            "input": inp,
            "output": out,
        }) for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"],
        )]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)

    # ── 4. 学習設定 ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,           # エポック数
        per_device_train_batch_size=1, # CPUの場合は1
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, # 実効バッチサイズ = 4
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",              # WandBなど不要
        use_cpu=True,                  # CPUで動かす
    )

    # ── 5. 学習実行 ──────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\n学習開始...")
    print("（CPUで動かす場合、数十分〜数時間かかります）")
    trainer.train()

    # ── 6. モデルの保存 ──────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nモデルを保存しました: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
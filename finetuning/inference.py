# finetuning/inference.py
"""
Fine-tuningしたモデルで推論する

ベースモデルとFine-tuningモデルの回答を比較します。
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "microsoft/phi-2"
LORA_DIR = "finetuning/lora_output"


def load_base_model():
    """ベースモデルを読み込む"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    return tokenizer, model


def load_finetuned_model():
    """Fine-tuningしたモデルを読み込む"""
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    # LoRAの重みを読み込む
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    return tokenizer, model


def generate(tokenizer, model, instruction: str, max_new_tokens: int = 200) -> str:
    """テキストを生成する"""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 入力部分を除いた生成テキストだけを返す
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    test_questions = [
        "F1スコアとは何ですか？",
        "Pandasで欠損値を処理する方法は？",
    ]

    print("=== ベースモデル vs Fine-tuningモデル 比較 ===\n")

    print("ベースモデルを読み込み中...")
    base_tokenizer, base_model = load_base_model()

    print("Fine-tuningモデルを読み込み中...")
    ft_tokenizer, ft_model = load_finetuned_model()

    for question in test_questions:
        print(f"\n質問: {question}")
        print("-" * 50)

        base_answer = generate(base_tokenizer, base_model, question)
        print(f"【ベースモデル】\n{base_answer}")

        ft_answer = generate(ft_tokenizer, ft_model, question)
        print(f"\n【Fine-tuningモデル】\n{ft_answer}")
        print()
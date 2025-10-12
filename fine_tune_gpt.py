import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import os
import csv
import math

SPECIAL_TOKENS = ["<|user|>", "<|bot|>", "<|endofbot|>"]

def build_pairs(example):
    turns = example["dialog"]
    pairs = []
    # Assume even idx ~ user, odd idx ~ bot (DailyDialog format)
    for u, b in zip(turns[::2], turns[1::2]):
        prompt = f"<|user|> {u}\n<|bot|>"
        response = f" {b} <|endofbot|>"
        pairs.append({"prompt": prompt, "response": response})
    return pairs

def flatten_dialogs(ds_split):
    rows = []
    for ex in ds_split:
        rows.extend(build_pairs(ex))
    return Dataset.from_list(rows)

def preprocess_fn(example, tok, max_len):
    text = example["prompt"] + example["response"]
    enc = tok(text, truncation=True, max_length=max_len, padding = "max_length")
    # mask prompt tokens -> ignore loss with -100
    prompt_ids = tok(example["prompt"], truncation=True, max_length=max_len, padding = "max_length")["input_ids"]
    labels = enc["input_ids"][:]
    for i in range(len(prompt_ids)):
        labels[i] = -100
    enc["labels"] = labels
    return enc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--out_dir", default="gpt2-sft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bsz", type=int, default=2, help="per-device train batch size")
    ap.add_argument("--eval_bsz", type=int, default=2)
    ap.add_argument("--grad_acc", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--packing", action="store_true", help="pack multiple samples per sequence")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lora", action="store_true", help="use LoRA adapters")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    # 1) Load DailyDialog
    raw = load_dataset("daily_dialog", trust_remote_code=True)
    train_pairs = flatten_dialogs(raw["train"])
    val_pairs   = flatten_dialogs(raw["validation"])

    # 2) Tokenizer with special tokens
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # 3) Tokenize & build labels
    def _prep(ex): return preprocess_fn(ex, tok, args.max_len)
    train_tok = train_pairs.map(_prep, remove_columns=train_pairs.column_names)
    val_tok   = val_pairs.map(_prep,   remove_columns=val_pairs.column_names)

    # 4) Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tok))

    # (optional) LoRA
    if args.lora:
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["c_attn","c_fc","c_proj"]  # GPT-2 blocks
        )
        model = get_peft_model(model, lcfg)

    # 5) Train setup
    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.eval_bsz,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=50,
        logging_strategy="steps",
        save_total_limit=2,
        report_to="tensorboard",
        fp16=args.fp16,
        bf16=args.bf16
    )

    def compute_metrics(eval_pred):
        _, metrics = eval_pred
        metrics["perplexity"] = math.exp(metrics["eval_loss"])
        return metrics
        
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        args=targs,
        packing=args.packing,
        compute_metrics=compute_metrics 
    )

    # 6) Train & save
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    # 7) Quick sample generation
    try:
        from transformers import pipeline
        pipe = pipeline("text-generation", model=args.out_dir, tokenizer=args.out_dir)
        prompt = "<|user|> hey, can you help me plan my day?\n<|bot|>"
        out = pipe(prompt, max_length=128, do_sample=True, top_p=0.9, temperature=0.8)[0]["generated_text"]
        print("\n=== SAMPLE ===\n", out)
    except Exception as e:
        print("Sample generation skipped:", e)

if __name__ == "__main__":
    main()

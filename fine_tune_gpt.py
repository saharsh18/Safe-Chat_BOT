import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

SPECIAL_TOKENS = {
    "user": "<|user|>",
    "bot": "<|bot|>",
    "end": "<|endofbot|>",
}

def create_chat_prompt(example):
    """
    Formats a single example from the daily_dialog dataset into the 'messages'
    format, which is a list of dictionaries with 'role' and 'content'.
    """
    messages = []
    # Assumes even turns are user, odd turns are bot
    for i, utterance in enumerate(example["dialog"]):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": utterance})
    return {"messages": messages}

def main():
    ap = argparse.ArgumentParser(description="Fine-tune a GPT-2 model for daily conversation.")
    ap.add_argument("--model_name", default="gpt2", help="The name of the base model to fine-tune.")
    ap.add_argument("--out_dir", default="gpt2-fine-tuned", help="Directory to save the fine-tuned model.")
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer.")
    ap.add_argument("--bsz", type=int, default=2, help="Per-device train batch size.")
    ap.add_argument("--grad_acc", type=int, default=8, help="Gradient accumulation steps.")
    ap.add_argument("--max_len", type=int, default=512, help="Maximum sequence length.")
    ap.add_argument("--save_steps", type=int, default=1000, help="Number of steps between model saves.")
    ap.add_argument("--packing", action="store_true", help="Pack multiple samples per sequence.")
    ap.add_argument("--lora", action="store_true", help="Use LoRA adapters.")
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    ap.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    args = ap.parse_args()

    # 1) Load DailyDialog
    print("Loading DailyDialog dataset...")
    raw = load_dataset("daily_dialog", trust_remote_code=True, split="train")
    dataset = raw.map(create_chat_prompt, remove_columns=raw.column_names)

    # 2) Tokenizer with special tokens
    print(f"Loading tokenizer from '{args.model_name}'...")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS.values())})

    tok.chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
                "{{'<|user|>' + message['content'] + '\n'}}"
            "{% elif message['role'] == 'assistant' %}"
                "{{'<|bot|>' + message['content'] + '<|endofbot|>' + '\n'}}"
            "{% endif %}"
        "{% endfor %}"
    )



    def formatting_func(batch):
        return [tok.apply_chat_template(messages, tokenize=False) for messages in batch["messages"]]

    # 3) Load model
    print(f"Loading model from '{args.model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tok))

    # (optional) LoRA
    if args.lora:
        lcfg = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            bias="none", 
            task_type="CAUSAL_LM",
            target_modules=["c_attn","c_fc","c_proj"] 
        )
        model = get_peft_model(model, lcfg)

    # 4) Setup Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # 5) Train setup
    targs = SFTConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        max_seq_length=args.max_len,
        num_train_epochs=args.epochs,
        eval_strategy="no",
        save_steps=args.save_steps,
        logging_steps=50,
        logging_strategy="steps",
        save_total_limit=2,
        report_to="tensorboard",
    )
        
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=dataset,
        args=targs,
        packing=args.packing,
        data_collator=data_collator,
        formatting_func=formatting_func,
    )

    # 6) Train & save
    print("Starting training...")
    trainer.train()

    print(f"Saving model to '{args.out_dir}'...")
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Model training and saving complete.")


if __name__ == "__main__":
    main()

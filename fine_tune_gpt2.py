import argparse
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig
from torch.utils.data import Subset
from collections import defaultdict
import random
from datasets import concatenate_datasets

SPECIAL_TOKENS = {
    "user": "<|user|>",
    "bot": "<|bot|>",
    "end": "<|endofbot|>",
}

# --------------------------
# Step 1: Load DailyDialog
# --------------------------
def process_daily_dialog():
    raw = load_dataset("daily_dialog", split="train")
    def create_chat_prompt(example):
        messages = []
        for i, utt in enumerate(example["dialog"]):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": utt})
        return {"messages": messages}

    data = raw.map(create_chat_prompt, remove_columns=raw.column_names)
    return data


# --------------------------
# Step 2: Load OASST1
# --------------------------
def process_oasst1(max_pairs=10000):
    """
    Loads and formats the OpenAssistant/oasst1 dataset.
    This version correctly filters for top-ranked English pairs.
    """
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    trees = defaultdict(list)
    for msg in ds:
        trees[msg["message_tree_id"]].append(msg)
    
    for t in trees:
        trees[t] = sorted(trees[t], key=lambda x: x["created_date"])

    pairs = []
    for msgs in trees.values():
        for i in range(len(msgs) - 1):
            prompter_msg = msgs[i]
            assistant_msg = msgs[i + 1]
            
            # --- UPDATED FILTER LOGIC ---
            if (prompter_msg["role"] == "prompter" and 
                assistant_msg["role"] == "assistant" and
                prompter_msg["lang"] == "en" and 
                assistant_msg["lang"] == "en" and
                assistant_msg.get("rank", -1) == 0): # Check for rank 0 (best answer)
                
                pairs.append({
                    "messages": [
                        {"role": "user", "content": prompter_msg["text"]},
                        {"role": "assistant", "content": assistant_msg["text"]}
                    ]
                })
    
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]
    
    # Add this check to be sure
    if not pairs:
        print("Warning: No pairs found for OASST1 with the new filter. Check dataset or filter logic.")
        
    return Dataset.from_list(pairs)


# --------------------------
# Step 3: Merge both datasets
# --------------------------
def combine_datasets():
    """Combines DailyDialog and OASST1."""
    d1 = process_daily_dialog()
    d2 = process_oasst1()
    print(f"DailyDialog samples: {len(d1)}, OASST1 samples: {len(d2)}")
    
    # Use the efficient concatenate_datasets function
    combined = concatenate_datasets([d1, d2]).shuffle(seed=42)
    
    print(f"Combined total samples: {len(combined)}")
    return combined


# --------------------------
# Step 4: Prepare tokenizer
# --------------------------
def prepare_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS.values())})

    # Chat template: convert structured messages to text
    tok.chat_template = (
        "{% for m in messages %}"
        "{% if m['role']=='user' %}{{'<|user|>' + m['content'] + '\\n'}}"
        "{% elif m['role']=='assistant' %}{{'<|bot|>' + m['content'] + '<|endofbot|>' + '\\n'}}"
        "{% endif %}"
        "{% endfor %}"
    )
    return tok


# --------------------------
# Step 5: Formatting function
# --------------------------
def formatting_func(batch):
    return [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in batch["messages"]]

# --------------------------
# Step 6: Train
# --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt2-medium")
    ap.add_argument("--out_dir", default="gpt2-medium-chatbot-finetuned")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad_acc", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--save_steps", type=int, default=1000)
    args = ap.parse_args()

    print("Preparing dataset...")
    dataset = combine_datasets()

    print("Loading tokenizer...")
    global tokenizer
    tokenizer = prepare_tokenizer(args.model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))


    targs = SFTConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        max_seq_length=args.max_len,
        num_train_epochs=args.epochs,
        evaluation_strategy="no",
        save_steps=args.save_steps,
        logging_steps=50,
        save_total_limit=2,
        report_to=["tensorboard"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=targs,
        formatting_func=formatting_func,
        packing=False,
    )

    print("Starting training...")
    trainer.train()
    print("Saving model...")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("✅ Training complete.")

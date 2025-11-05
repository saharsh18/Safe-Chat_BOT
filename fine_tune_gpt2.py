import argparse
import os
import random
from collections import defaultdict
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig
import torch 
import numpy as np

# Special Tokens

SPECIAL_TOKENS = {
    "user": "<|user|>",
    "bot": "<|bot|>",
    "end": "<|endofbot|>",
}

def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# -----------------------------------

# ============================================================
# Step 1: Process DailyDialog
# ============================================================
def process_daily_dialog(limit=10000): # Limit samples
    print(f"Loading DailyDialog (limit {limit})...")
    raw = load_dataset("daily_dialog", split="train")
    if len(raw) > limit:
        raw = raw.shuffle(seed=42).select(range(limit))

    def create_chat_prompt(example):
        messages = []
        for i, utt in enumerate(example["dialog"]):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": utt})
        return {"messages": messages}

    data = raw.map(create_chat_prompt, remove_columns=raw.column_names)
    return data

# ============================================================
# Step 2: Process OASST1
# ============================================================
def build_tree(messages):
    children = defaultdict(list)
    for msg in messages:
        children[msg.get("parent_id")].append(msg)
    return children

def dfs_paths(children, node, path=None):
    if path is None: path = []
    path = path + [node]
    if node["message_id"] not in children or not children[node["message_id"]]:
        yield path
    else:
        for child in children[node["message_id"]]:
            yield from dfs_paths(children, child, path)

def process_oasst1(max_convos=10000, min_turns=2, max_turns=10):
    print(f"Loading OASST1 (limit {max_convos})...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    trees = defaultdict(list)
    for msg in ds:
        trees[msg["message_tree_id"]].append(msg)

    all_conversations = []
    for msgs in trees.values():
        if len(all_conversations) >= max_convos:
             break # Stop processing new trees if we hit the limit
             
        msgs = sorted(msgs, key=lambda x: x["created_date"])
        children = build_tree(msgs)
        roots = [m for m in msgs if m.get("parent_id") is None]

        for root in roots:
            if len(all_conversations) >= max_convos:
                 break
                 
            for path in dfs_paths(children, root):
                if len(all_conversations) >= max_convos:
                     break
                     
                if not all(p.get("lang", "") == "en" for p in path):
                    continue

                convo = []
                for p in path:
                    if p["role"] == "prompter":
                        convo.append({"role": "user", "content": p["text"]})
                    elif p["role"] == "assistant":
                        convo.append({"role": "assistant", "content": p["text"]})

                if len(convo) < min_turns or len(convo) > max_turns:
                    continue

                all_conversations.append({"messages": convo})

    random.shuffle(all_conversations)
    all_conversations = all_conversations[:max_convos]

    print(f" Extracted {len(all_conversations)} valid multi-turn conversations from OASST1.")
    return Dataset.from_list(all_conversations)

# ============================================================
# Step 3: Process Anthropic HH (The Missing Piece)
# ============================================================
def process_anthropic_hh(limit=5000):
    print(f"Loading Anthropic HH (limit {limit})...")
    all_hh_convos = []
    
    try:
        for subset in ["harmless-base", "helpful-base"]:
            ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="train")
            for ex in ds:
                raw_text = ex.get("chosen", "")
                if not raw_text:
                    continue
                    
                turns = raw_text.strip().split("\n\n")
                messages = []
                for turn in turns:
                    if turn.startswith("Human:"):
                        messages.append({"role": "user", "content": turn.replace("Human:", "").strip()})
                    elif turn.startswith("Assistant:"):
                        messages.append({"role": "assistant", "content": turn.replace("Assistant:", "").strip()})
                
                if messages:
                    all_hh_convos.append({"messages": messages})

        random.shuffle(all_hh_convos)
        if len(all_hh_convos) > limit:
            all_hh_convos = all_hh_convos[:limit]
            
        print(f" Extracted {len(all_hh_convos)} valid conversations from Anthropic HH.")
        return Dataset.from_list(all_hh_convos)
        
    except Exception as e:
        print(f"Could not load Anthropic HH: {e}. Skipping.")
        return None

# ============================================================
# Step 4: Combine All Datasets
# ============================================================
def combine_datasets(): 
    print("Processing DailyDialog...")
    d1 = process_daily_dialog(limit=12500) 

    print("Processing OASST1...")
    d2 = process_oasst1(max_convos=10000)

    # print("Processing Anthropic HH...")
    # d3 = process_anthropic_hh(limit=5000)

    datasets_to_mix = [d for d in [d1, d2] if d is not None]
    if not datasets_to_mix:
        raise ValueError("No datasets could be loaded!")
        
    # Check lengths safely
    len_d1 = len(d1) if d1 else 0
    len_d2 = len(d2) if d2 else 0
    # len_d3 = len(d3) if d3 else 0
        
    print(f"Samples: DailyDialog={len_d1}, OASST1={len_d2}")
    combined = concatenate_datasets(datasets_to_mix).shuffle(seed=42)
    print(f"Combined total samples: {len(combined)}")
    return combined

# ============================================================
# Step 5: Tokenizer Setup
# ============================================================
def prepare_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS.values())})

    tok.chat_template = (
        "{% for m in messages %}"
            "{% if m['role']=='user' %}"
                "{{ '<|user|>' + m['content'] + '\\n' }}"
            "{% elif m['role']=='assistant' %}"
                "{{ '<|bot|>' + m['content'] + '<|endofbot|>' + '\\n' }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{ '<|bot|>' }}"
        "{% endif %}"
    )
    return tok

# ============================================================
# Step 6: Formatting Function for Trainer
# ============================================================
def formatting_func(batch):
    # This function now correctly applies the template *once* per conversation
    # SFTTrainer will handle tokenization and packing.
    return [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in batch["messages"]]

# ============================================================
# Step 7: Main Training Loop
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt2-medium")
    ap.add_argument("--out_dir", default="gpt2-medium-sft-mixed-stable-not-safe-2") 
    ap.add_argument("--epochs", type=int, default=3) 
    ap.add_argument("--lr", type=float, default=2e-5) 
    ap.add_argument("--bsz", type=int, default=2) 
    ap.add_argument("--grad_acc", type=int, default=8) 
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--save_steps", type=int, default=1000)
    
    args = ap.parse_args()
    set_seed(42) 

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
        save_strategy="steps", 
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
    )

    print(" Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Training complete.")
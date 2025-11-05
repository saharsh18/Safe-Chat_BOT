import argparse
import torch
import gc
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from trl import RewardTrainer
import os
import random
import numpy as np

# --- Constants ---
DATASET_NAME = "Anthropic/hh-rlhf"
DATASET_SUBSETS = "harmless-base,helpful-base"
TRAIN_SPLIT = "train"
MAX_LENGTH = 512 

# --- Helper Functions ---

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_anthropic_hh(examples, tokenizer, max_length):
    """
    Tokenizes chosen and rejected texts for RewardTrainer.
    """
    new_examples = {
        "input_ids_chosen": [], "attention_mask_chosen": [],
        "input_ids_rejected": [], "attention_mask_rejected": []
    }

    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
       
        chosen_text = chosen.strip()
        rejected_text = rejected.strip()

        
        # The RewardTrainer's collator will handle padding dynamically.
        tokenized_chosen = tokenizer(chosen_text, truncation=True, max_length=max_length)
        tokenized_rejected = tokenizer(rejected_text, truncation=True, max_length=max_length)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Preference Model (Reward Model) on Anthropic HH-RLHF.")
    
    parser.add_argument("--base_model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--output_dir", type=str, default="preference-model-deberta-v3")
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Tokenizer ---
    print(f"Loading tokenizer: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"
        print(f"Set pad_token = {tokenizer.pad_token}")

    # --- 2. Load Dataset ---
    print(f"Loading dataset: {DATASET_NAME} ({DATASET_SUBSETS})...")
    subsets = [s.strip() for s in DATASET_SUBSETS.split(",")]
    train_datasets = []

    for sub in subsets:
        try:
            train_ds = load_dataset(DATASET_NAME, data_dir=sub, split=TRAIN_SPLIT, cache_dir=args.cache_dir)
            train_datasets.append(train_ds)
        except Exception as e:
            print(f"Could not load subset '{sub}': {e}")

    if not train_datasets:
        raise ValueError("No training data loaded.")

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=args.seed)
    print(f"Total train examples: {len(train_dataset)}")

    # --- 3. Preprocess ---
    print("Tokenizing train dataset...")
    train_dataset = train_dataset.map(
        preprocess_anthropic_hh,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=train_dataset.column_names,
    )

    train_dataset = train_dataset.filter(lambda x: x['input_ids_chosen'] != x['input_ids_rejected'])
    print(f"Filtered train examples: {len(train_dataset)}")

    # --- 4. Load Model ---
    print(f"Loading base model: {args.base_model}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, cache_dir=args.cache_dir
    )

    # --- 5. Training Args ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="no", 
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="tensorboard",
        remove_unused_columns=False,
        seed=args.seed,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    # --- 6. Trainer ---
    print("Initializing RewardTrainer...")
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        max_length=MAX_LENGTH 
    )

    # --- 7. Train ---
    print(" Starting Preference Model training...")
    trainer.train()
    print(" Training complete.")

    # --- 8. Save ---
    print(f" Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(" Model saved successfully.")
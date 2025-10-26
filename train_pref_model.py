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
# Dataset Configuration
DATASET_NAME = "Anthropic/hh-rlhf"
DATASET_SUBSETS = "harmless-base,helpful-base" # Using both harmless and helpful
TRAIN_SPLIT = "train"
MAX_LENGTH = 512 # Max sequence length for tokenizer during PM training

# --- Helper Functions ---

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_anthropic_hh(examples, tokenizer, max_length):
    """
    Tokenizes the chosen and rejected texts for the RewardTrainer.
    """
    new_examples = {
        "input_ids_chosen": [], "attention_mask_chosen": [],
        "input_ids_rejected": [], "attention_mask_rejected": []
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, truncation=True, max_length=max_length)
        tokenized_rejected = tokenizer(rejected, truncation=True, max_length=max_length)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

# --- Main Training Function ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Preference Model (Reward Model) on Anthropic HH-RLHF.")
    # Model & Output Args
    parser.add_argument("--base_model", type=str, default="distilbert-base-uncased", help="Base model name or path (e.g., distilbert, gpt2).")
    parser.add_argument("--output_dir", type=str, default="preference-model", help="Directory to save the trained preference model.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face cache directory.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Train batch size per GPU (adjust based on VRAM).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log training metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- GPU Check ---
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training on CPU. This will be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"CUDA available. Using {torch.cuda.get_device_name(0)}")

    # --- 1. Load Tokenizer ---
    print(f"Loading tokenizer: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    if "gpt2" in args.base_model.lower() and tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
         print("Set pad_token to eos_token for GPT2-based model.")

    # --- 2. Load and Process Dataset ---
    print(f"Loading dataset: {DATASET_NAME} (subsets: {DATASET_SUBSETS})...")
    subsets_to_load = DATASET_SUBSETS.split(',')
    train_datasets = []

    for sub in subsets_to_load:
        print(f"  Loading subset: {sub.strip()}")
        try:
            if "/" in DATASET_NAME: # Needs data_dir
                train_ds = load_dataset(DATASET_NAME, data_dir=sub.strip(), split=TRAIN_SPLIT, cache_dir=args.cache_dir)
            else: 
                train_ds = load_dataset(DATASET_NAME, name=sub.strip(), split=TRAIN_SPLIT, cache_dir=args.cache_dir)
            train_datasets.append(train_ds)
        except Exception as e:
            print(f"  Warning: Could not load subset '{sub.strip()}': {e}")

    # Check only train_datasets
    if not train_datasets:
        raise ValueError("Failed to load any training dataset subsets specified in constants.")

    train_dataset = concatenate_datasets(train_datasets)
    print(f"Total train examples: {len(train_dataset)}")

    train_dataset = train_dataset.shuffle(seed=args.seed)

    print("Preprocessing dataset...")
    train_dataset = train_dataset.map(
        preprocess_anthropic_hh, batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=train_dataset.column_names
    )

    # Filter out examples where chosen/rejected are identical after tokenization
    train_dataset = train_dataset.filter(lambda x: x['input_ids_chosen'] != x['input_ids_rejected'])
    print(f"Filtered train examples: {len(train_dataset)}")

    # --- 3. Load Base Model for Reward Modeling ---
    print(f"Loading base model: {args.base_model}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, cache_dir=args.cache_dir,
    )

    # --- 4. Configure Training ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available(), 
        evaluation_strategy="no",
        save_strategy="steps", # Keep saving checkpoints based on steps
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="tensorboard",
        remove_unused_columns=False,
        seed=args.seed,
    )

    # --- 5. Initialize RewardTrainer ---
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
    )

    # --- 6. Train ---
    print("Starting Preference Model training...")
    trainer.train()
    print("Training complete.")

    # --- 7. Save Final Model ---
    print(f"Saving final preference model to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Preference model saved successfully.")
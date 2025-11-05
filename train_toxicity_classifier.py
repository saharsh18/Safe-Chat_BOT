import argparse
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import random
import gc

# --- Constants ---
MODEL_NAME = "distilbert-base-uncased"
CIVIL_DATASET = "civil_comments"
CIVIL_TEXT_COL = "text"
TOXICITY_COLUMNS = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']  

# --- Data Processing Functions ---
def process_civil_comments_data():
    """
    Loads civil_comments data and processes it for regression.
    The label will be the *maximum* score across all 7 toxicity categories.
    """
    print(f"Loading civil_comments dataset")
    try:
        dataset = load_dataset(CIVIL_DATASET, split="train")
        
        def is_valid(example):
            if example[CIVIL_TEXT_COL] is None:
                return False
            for col in TOXICITY_COLUMNS:
                if example[col] is None or not isinstance(example[col], float):
                    return False
            return True
            
        dataset = dataset.filter(is_valid)
        dataset = dataset.shuffle(seed=42)
    except Exception as e:
        print(f"Could not load civil_comments dataset: {e}. Skipping.")
        return None
    

    def create_regression_label(example):
        scores = [float(example[col]) for col in TOXICITY_COLUMNS]
        max_score = np.sum(scores)
        return {
            "text": example[CIVIL_TEXT_COL],
            "label": max_score 
        }
        

    dataset = dataset.map(create_regression_label, remove_columns=dataset.column_names)
    print(f"Processed {len(dataset)} civil_comments samples using MAX score from 7 columns.")
    return dataset

# --- Main Training Function ---
def train_classifier(args):
    """
    Main function to load data, train the classifier, and save it.
    """
    # --- 1. Load and Combine Datasets ---
    print("Loading and processing datasets...")
    civil_data = process_civil_comments_data() 

    train_dataset = civil_data.shuffle(seed=42)

    del civil_data
    gc.collect()

    print(f"Total Train samples: {len(train_dataset)}")

    # --- 2. Load Tokenizer ---
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Preprocessing Function ---
    def preprocess_function(examples):
        tokenized_batch = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_len
        )
        tokenized_batch["label"] = [float(label) for label in examples["label"]]
        return tokenized_batch

    print("Tokenizing dataset (this may take a moment)...")
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text"])

    # --- 4. Load Model (for REGRESSION) ---
    print(f"Loading model: {MODEL_NAME} for regression...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1  # CRITICAL: num_labels=1 for regression
    )

    # --- 5. Set Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_acc,
        
        fp16=torch.cuda.is_available(), 
        
        evaluation_strategy="no",     
        save_strategy="epoch",        
        save_total_limit=2,           
        
        logging_steps=50,
        report_to="tensorboard",
    )

    # --- 6. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # --- 7. Train ---
    print("Starting classifier training...")
    trainer.train()
    print("Training complete.")

    # --- 8. Save Final Model ---
    print(f"Saving final classifier model to '{args.out_dir}'...")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Classifier saved successfully.")

# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a toxicity/undesirability classifier.")
    parser.add_argument("--out_dir", type=str, default="toxicity-classifier2", help="Directory to save the model.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size (per device).")
    parser.add_argument("--grad_acc", type=int, default=2, help="Gradient accumulation steps (effective batch size = 16*2=32).")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length for tokenizer.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training on CPU. This will be very slow.")
    else:
        print(f"CUDA available. Using {torch.cuda.get_device_name(0)}")
        
    train_classifier(args)
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments 
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
import datetime
import math

# --- Constants ---
MODEL_NAME = "distilbert-base-uncased"
CIVIL_DATASET = "civil_comments"
CIVIL_TEXT_COL = "text"
TOXICITY_COLUMNS = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']
MAX_TRUE_LABEL = 7.0 
ACCURACY_TOLERANCE = 0.5 

# --- Helper Functions ---

def process_civil_comments_data(split="test"):
    """
    Loads civil_comments data and processes it for regression (SUM of toxicity scores).
    Returns the processed Hugging Face Dataset object.
    """
    print(f"Loading civil_comments dataset split: {split}")
    try:
        dataset = load_dataset(CIVIL_DATASET, split=split)
        
        def is_valid(example):
            if example[CIVIL_TEXT_COL] is None:
                return False
            for col in TOXICITY_COLUMNS:
                if example.get(col) is None or not isinstance(example[col], (int, float, np.float32)):
                    return False
            return True
            
        dataset = dataset.filter(is_valid)
    except Exception as e:
        print(f"Could not load civil_comments dataset: {e}. Exiting.")
        return None

    def create_regression_label(example):
        # The label is the SUM of all 7 scores (0.0 to 7.0)
        scores = [float(example[col]) for col in TOXICITY_COLUMNS]
        sum_score = np.sum(scores)
        return {
            "text": example[CIVIL_TEXT_COL],
            "label": sum_score
        }
        
    dataset = dataset.map(create_regression_label, remove_columns=dataset.column_names)
    print(f"Processed {len(dataset)} {split} samples using SUM score.")
    return dataset

def compute_metrics(eval_preds):
    """
    Custom function to compute regression metrics, including R-squared,
    Max Absolute Error, CV(RMSE), and Thresholded Accuracy.
    """
    predictions, labels = eval_preds
    predictions = predictions.flatten() 
    
    # 1. Standard Error Metrics
    mse = mean_squared_error(labels, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)

    # 2. R-squared (Coefficient of Determination)
    r2 = r2_score(labels, predictions)
    
    # 3. Max Absolute Error (Worst-Case Error)
    max_ae = np.max(np.abs(labels - predictions))
    
    # 4. Coefficient of Variation of the RMSE (CV(RMSE))
    labels_mean = np.mean(labels)
    cv_rmse = (rmse / labels_mean) * 100 if labels_mean != 0 else 0.0

    # 5. Thresholded Accuracy (The new classification-like metric)
    # Checks if the absolute error is less than or equal to the defined tolerance
    is_accurate = np.abs(labels - predictions) <= ACCURACY_TOLERANCE
    threshold_accuracy = np.mean(is_accurate) # Mean of booleans gives the percentage
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "r2_score": r2,
        "max_ae": max_ae,
        "cv_rmse_percent": cv_rmse,
        "thresholded_accuracy": threshold_accuracy
    }

# --- Main Evaluation Function ---

def evaluate_model(args):
    """
    Loads a trained toxicity regression model and performs quantitative evaluation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Tokenizer and Model ---
    try:
        print(f"Loading tokenizer and model from: {args.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=1) 
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # --- 2. Load and Preprocess Data ---
    full_eval_dataset = process_civil_comments_data(split="test") 
    if full_eval_dataset is None:
        return

    def preprocess_function(examples):
        tokenized_batch = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_len
        )
        tokenized_batch["label"] = [float(label) for label in examples["label"]]
        return tokenized_batch

    print("Tokenizing evaluation dataset...")
    tokenized_dataset = full_eval_dataset.map(preprocess_function, batched=True)
    
    tokenized_dataset = tokenized_dataset.remove_columns([col for col in full_eval_dataset.column_names if col not in ['input_ids', 'attention_mask', 'label']])
    
    if args.limit and args.limit < len(tokenized_dataset):
        tokenized_dataset = tokenized_dataset.shuffle(seed=42).select(range(args.limit))
        print(f"Limiting evaluation to {len(tokenized_dataset)} samples.")
    
    # --- 3. Run Evaluation to get Quantitative Metrics ---
    
    training_args = TrainingArguments(
        output_dir="tmp_eval_out",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        do_train=False,
        do_eval=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Starting quantitative evaluation...")
    metrics = trainer.evaluate()
    print("Quantitative evaluation complete.")
    
    # --- 4. Save Results to JSON ---
    evaluation_results = {
        "metadata": {
            "model_dir": args.model_dir,
            "base_model": MODEL_NAME,
            "device": device,
            "max_len": args.max_len,
            "evaluation_timestamp": datetime.datetime.now().isoformat(),
            "total_samples": len(tokenized_dataset),
            "max_possible_label": MAX_TRUE_LABEL,
            "accuracy_tolerance": ACCURACY_TOLERANCE
        },
        "quantitative_metrics": metrics
    }

    output_file = args.output_file
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    print(f"\nSaving evaluation results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"\n Evaluation complete. Results saved to {output_file}")
    
    # Display Key Results
    print("\n--- Key Quantitative Metrics (Expanded) ---")
    print(f"RMSE (Root Mean Squared Error): {metrics.get('eval_rmse', 'N/A'):.4f}")
    print(f"MAE (Mean Absolute Error): {metrics.get('eval_mae', 'N/A'):.4f}")
    print(f"R-squared (R² Score): {metrics.get('eval_r2_score', 'N/A'):.4f}")
    print(f"Max Absolute Error (Worst Case): {metrics.get('eval_max_ae', 'N/A'):.4f}")
    print(f"Thresholded Accuracy (Tolerance +/-{ACCURACY_TOLERANCE}): {metrics.get('eval_thresholded_accuracy', 'N/A'):.2f}")


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned toxicity regression classifier.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the trained model is saved.")
    parser.add_argument("--output_file", type=str, default="results/toxicity_model_eval_metrics.json", help="File to save evaluation results in JSON format.")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size (per device).")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length (must match training).")
    parser.add_argument("--limit", type=int, default=10000, help="Limit the number of evaluation samples.")
    args = parser.parse_args()
        
    evaluate_model(args)
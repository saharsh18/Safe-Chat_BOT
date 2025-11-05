import argparse
import torch
import gc
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import random
import numpy as np
from tqdm import tqdm
import json
import torch.nn.functional as F

# --- Constants ---
DATASET_NAME = "Anthropic/hh-rlhf"
DATASET_SUBSETS = "harmless-base,helpful-base"
EVAL_SPLIT = "test"
MAX_LENGTH = 512

# --- Helper Functions ---
def set_seed(seed):
    """Sets the random seed for an experiment for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_anthropic_hh(examples, tokenizer, max_length):
    """
    Tokenize chosen and rejected samples.
    We pad to max_length here for simplified batching in the manual loop.
    """
    new_examples = {
        "input_ids_chosen": [], "attention_mask_chosen": [],
        "input_ids_rejected": [], "attention_mask_rejected": []
    }

    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        chosen_text = chosen.strip()
        rejected_text = rejected.strip()

        # Tokenize both examples; pad to max_length for batching
        tokenized_chosen = tokenizer(
            chosen_text, truncation=True, max_length=max_length, padding="max_length"
        )
        tokenized_rejected = tokenizer(
            rejected_text, truncation=True, max_length=max_length, padding="max_length"
        )

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


# --- Main Evaluation ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Preference Model (Reward Model).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained preference model.")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save metrics.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max number of samples to evaluate on.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 for inference if available.")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model & Tokenizer ---
    print(f"Loading model and tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.use_bf16 and torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()

    # --- 2. Load Evaluation Data ---
    print(f"Loading dataset: {DATASET_NAME} (subsets: {DATASET_SUBSETS})...")
    subsets = [s.strip() for s in DATASET_SUBSETS.split(",")]
    eval_datasets = []

    for sub in subsets:
        try:
            ds = load_dataset(DATASET_NAME, data_dir=sub, split=EVAL_SPLIT)
            eval_datasets.append(ds)
        except Exception as e:
            print(f"Could not load subset '{sub}': {e}")

    if not eval_datasets:
        raise ValueError("No evaluation subsets loaded.")

    eval_dataset = concatenate_datasets(eval_datasets)
    print(f"Loaded {len(eval_dataset)} total evaluation samples.")

    if args.max_eval_samples:
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(args.max_eval_samples))
        print(f"Evaluating on subset of {len(eval_dataset)} samples.")

    # --- 3. Tokenize ---
    print("Tokenizing evaluation dataset...")
    eval_dataset = eval_dataset.map(
        preprocess_anthropic_hh,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=eval_dataset.column_names,
    )
    eval_dataset = eval_dataset.filter(lambda x: x["input_ids_chosen"] != x["input_ids_rejected"])
    eval_dataset.set_format(type="torch", columns=[
        "input_ids_chosen", "attention_mask_chosen",
        "input_ids_rejected", "attention_mask_rejected"
    ])

    print(f"Final evaluation set size: {len(eval_dataset)}")

    # --- 4. Evaluation Loop ---
    print("Running preference model evaluation...")
    total_correct = 0
    total_pairs = 0
    all_scores_chosen = []
    all_scores_rejected = []
    all_batch_losses = [] # To store the loss for each batch

    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs_chosen = {
                "input_ids": batch["input_ids_chosen"].to(device),
                "attention_mask": batch["attention_mask_chosen"].to(device),
            }
            inputs_rejected = {
                "input_ids": batch["input_ids_rejected"].to(device),
                "attention_mask": batch["attention_mask_rejected"].to(device),
            }

            scores_chosen_gpu = model(**inputs_chosen).logits
            scores_rejected_gpu = model(**inputs_rejected).logits

            # --- Calculate Loss (on GPU) ---
            loss_tensor = -F.logsigmoid(scores_chosen_gpu - scores_rejected_gpu)
            mean_loss = loss_tensor.mean().item()
            all_batch_losses.append(mean_loss)

            # --- Calculate Accuracy (on CPU) ---
            scores_chosen = scores_chosen_gpu.squeeze().cpu()
            scores_rejected = scores_rejected_gpu.squeeze().cpu()

            # Handle single-element batches
            if scores_chosen.dim() == 0: scores_chosen = scores_chosen.unsqueeze(0)
            if scores_rejected.dim() == 0: scores_rejected = scores_rejected.unsqueeze(0)

            # This is the "eval_accuracy" from your first script
            total_correct += (scores_chosen > scores_rejected).sum().item()
            total_pairs += len(scores_chosen)

            # These are the metrics from your second script
            all_scores_chosen.extend(scores_chosen.tolist())
            all_scores_rejected.extend(scores_rejected.tolist())

            # Clean up memory
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --- 5. Calculate Final Metrics ---
    accuracy = (total_correct / total_pairs) * 100 if total_pairs > 0 else 0
    avg_loss = np.mean(all_batch_losses) if all_batch_losses else 0
    avg_score_chosen = np.mean(all_scores_chosen) if all_scores_chosen else 0
    avg_score_rejected = np.mean(all_scores_rejected) if all_scores_rejected else 0
    avg_score_diff = avg_score_chosen - avg_score_rejected

    # --- 6. Store Metrics in Dictionary ---
    metrics = {
        "eval_accuracy": accuracy,
        "eval_loss": avg_loss,
        "eval_avg_score_chosen": avg_score_chosen,
        "eval_avg_score_rejected": avg_score_rejected,
        "eval_avg_score_diff": avg_score_diff,
        "total_pairs": total_pairs,
        "total_correct": total_correct,
        "model_path": args.model_path,
        "max_eval_samples": args.max_eval_samples if args.max_eval_samples else total_pairs
    }

    # --- 7. Print Results to Terminal ---
    print("\n--- Preference Model Evaluation Results ---")
    print(f"Accuracy: {metrics['eval_accuracy']:.2f}% ({metrics['total_correct']}/{metrics['total_pairs']})")
    print(f"Eval Loss: {metrics['eval_loss']:.4f}")
    print(f"Avg Score (Chosen):   {metrics['eval_avg_score_chosen']:.4f}")
    print(f"Avg Score (Rejected): {metrics['eval_avg_score_rejected']:.4f}")
    print(f"Avg Score Diff:       {metrics['eval_avg_score_diff']:.4f}")

    # --- 8. Save Metrics to File ---
    os.makedirs("results", exist_ok=True)
    metrics_path = os.path.join("results", "pref_model_eval_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

    # --- 9. Example Qualitative Outputs ---
    print("\n--- Example Predictions (First 5) ---")
    for i in range(min(5, total_pairs)):
        print(f"\nExample {i+1}:")
        print(f"  Chosen Score:   {all_scores_chosen[i]:.4f}")
        print(f"  Rejected Score: {all_scores_rejected[i]:.4f}")
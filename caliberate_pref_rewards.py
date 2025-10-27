import argparse
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from reward_utils import get_raw_preference_scores, set_seed # Import from our new file
import os

# --- Constants ---
# Use the same 'good' data as our preference model eval
DATASET_NAME = "Anthropic/hh-rlhf"
DATASET_SUBSETS = "harmless-base,helpful-base"
EVAL_SPLIT = "test"
MAX_LENGTH = 512 # Match PM max length

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the Preference Model (PM).")
    parser.add_argument("--pref_model_dir", type=str, required=True, help="Path to the trained preference model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for calibration.")
    
    args = parser.parse_args()
    set_seed()

    # --- GPU Check ---
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        exit()
    else:
        device = torch.device("cuda")
        print(f"CUDA available. Using {torch.cuda.get_device_name(0)}")

    # --- 1. Load Preference Model and Tokenizer ---
    print(f"Loading preference model from: {args.pref_model_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.pref_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pref_model_dir,
    ).to(device)
    model.eval()

    # --- 2. Load Calibration Dataset (Good Quality Responses) ---
    print(f"Loading dataset: {DATASET_NAME} (subsets: {DATASET_SUBSETS})...")
    subsets_to_load = DATASET_SUBSETS.split(',')
    eval_datasets = []

    for sub in subsets_to_load:
        print(f"  Loading subset: {sub.strip()} - {EVAL_SPLIT} split")
        try:
            split_to_load = EVAL_SPLIT
            if "/" in DATASET_NAME:
                eval_ds = load_dataset(DATASET_NAME, data_dir=sub.strip(), split=split_to_load)
            else:
                eval_ds = load_dataset(DATASET_NAME, name=sub.strip(), split=split_to_load)
            eval_datasets.append(eval_ds)
        except Exception as e:
            print(f"  Warning: Could not load subset '{sub.strip()}': {e}")

    if not eval_datasets:
        raise ValueError("Failed to load any evaluation dataset subsets.")

    eval_dataset = concatenate_datasets(eval_datasets)
    
    # We calibrate on the 'chosen' (good) responses
    calibration_texts = eval_dataset["chosen"]
    print(f"Loaded {len(calibration_texts)} 'chosen' responses for calibration.")

    # --- 3. Calculate Raw Scores ---
    # We use the function from our new reward_utils.py file
    raw_scores = get_raw_preference_scores(
        model, tokenizer, calibration_texts, 
        args.batch_size, device, MAX_LENGTH
    )
    
    # --- 4. Compute and Print Stats ---
    mean_score = np.mean(raw_scores)
    std_score = np.std(raw_scores)

    print("\n--- Preference Model Calibration Results ---")
    print(f"\nPREF_MEAN: {mean_score:.4f}")
    print(f"PREF_STD:  {std_score:.4f}")
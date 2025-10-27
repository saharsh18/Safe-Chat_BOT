import argparse
import torch
import numpy as np
from datasets import load_dataset # Only need load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# --- Import the correct function ---
from reward_utils import get_toxicity_scores, set_seed
import os

# --- Constants ---
# --- Use the dataset the toxicity model was trained on ---
DATASET_NAME = "civil_comments"
TEXT_COLUMN = "text" # The column containing the text data
EVAL_SPLIT = "test" # Use the test split for calibration consistency
# Limit calibration data for speed if needed
CALIBRATION_LIMIT = 10000
# --- Use the max length relevant for the toxicity model ---
MAX_LENGTH = 128

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the Toxicity Model (Safety Score) using civil_comments.")
    parser.add_argument("--tox_model_dir", type=str, required=True, help="Path to the trained toxicity classifier model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for calibration.") # Increased default batch size
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face cache directory.")

    args = parser.parse_args()
    set_seed()

    # --- GPU Check ---
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        exit()
    else:
        device = torch.device("cuda")
        print(f"CUDA available. Using {torch.cuda.get_device_name(0)}")

    # --- 1. Load Toxicity Model and Tokenizer ---
    print(f"Loading toxicity model from: {args.tox_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tox_model_dir, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set toxicity tokenizer pad_token to eos_token.")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.tox_model_dir, cache_dir=args.cache_dir
    ).to(device)
    model.eval()

    # --- 2. Load Calibration Dataset (civil_comments) ---
    print(f"Loading dataset: {DATASET_NAME} - {EVAL_SPLIT} split...")
    try:
        # Load the test split directly
        eval_dataset = load_dataset(DATASET_NAME, split=f"{EVAL_SPLIT}[:{CALIBRATION_LIMIT}]", cache_dir=args.cache_dir) # Limit for speed
        # Filter out potential None or non-string entries just in case
        eval_dataset = eval_dataset.filter(lambda x: x[TEXT_COLUMN] is not None and isinstance(x[TEXT_COLUMN], str))
        calibration_texts = eval_dataset[TEXT_COLUMN]
        print(f"Loaded {len(calibration_texts)} texts from {DATASET_NAME} for calibration.")
    except Exception as e:
        print(f"  Error loading {DATASET_NAME}: {e}")
        raise ValueError("Failed to load calibration dataset.")


    # --- 3. Calculate Safety Scores ---
    # Use get_toxicity_scores function on the loaded texts
    print("Calculating safety scores...")
    safety_scores = get_toxicity_scores(
        model, tokenizer, calibration_texts,
        args.batch_size, device, MAX_LENGTH
    )
    # ------------------------------------

    # --- 4. Compute and Print Stats ---
    mean_score = np.mean(safety_scores)
    std_score = np.std(safety_scores)

    print("\n--- Toxicity Model Calibration Results (Safety Scores on civil_comments) ---")
    print("These are stats for the (1.0 - prob_toxic) score.")
    print(f"\nTOX_MEAN (Safety): {mean_score:.4f}")
    print(f"TOX_STD (Safety):  {std_score:.4f}")
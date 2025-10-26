import argparse
import torch
import gc
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
import random
import numpy as np
from tqdm import tqdm

# --- Constants ---
# Use the same dataset config as the training script
DATASET_NAME = "Anthropic/hh-rlhf"
DATASET_SUBSETS = "harmless-base,helpful-base" # Should match training
EVAL_SPLIT = "test"
MAX_LENGTH = 512 # Should match training

# --- Helper Functions ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_anthropic_hh(examples, tokenizer, max_length):
    """
    Tokenizes the chosen and rejected texts.
    (Identical to the training script)
    """
    new_examples = {
        "input_ids_chosen": [], "attention_mask_chosen": [],
        "input_ids_rejected": [], "attention_mask_rejected": []
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, truncation=True, max_length=max_length, padding="max_length") # Pad here for batching
        tokenized_rejected = tokenizer(rejected, truncation=True, max_length=max_length, padding="max_length") # Pad here for batching

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

# --- Main Evaluation Function ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Preference Model (Reward Model).")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the preference model is saved.")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size (per device).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional: Max number of samples to evaluate (for quick check).")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 for evaluation (if model was trained with it).")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- GPU Check ---
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, evaluating on CPU. This will be slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"CUDA available. Using {torch.cuda.get_device_name(0)}")

    # --- 1. Load Model and Tokenizer ---
    print(f"Loading model and tokenizer from: {args.model_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
        ).to(device)
        model.eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        exit()

    # --- 2. Load and Process Evaluation Dataset ---
    print(f"Loading dataset: {DATASET_NAME} (subsets: {DATASET_SUBSETS})...")
    subsets_to_load = DATASET_SUBSETS.split(',')
    eval_datasets = []

    for sub in subsets_to_load:
        print(f"  Loading subset: {sub.strip()} - {EVAL_SPLIT} split")
        try:
            # Use the predefined TEST split
            split_to_load = EVAL_SPLIT
            if "/" in DATASET_NAME:
                eval_ds = load_dataset(DATASET_NAME, data_dir=sub.strip(), split=split_to_load)
            else:
                eval_ds = load_dataset(DATASET_NAME, name=sub.strip(), split=split_to_load)
            eval_datasets.append(eval_ds)
        except Exception as e:
            print(f"  Warning: Could not load subset '{sub.strip()}': {e}")

    if not eval_datasets:
        raise ValueError("Failed to load any evaluation dataset subsets specified in constants.")

    eval_dataset = concatenate_datasets(eval_datasets)
    print(f"Total eval examples loaded: {len(eval_dataset)}")

    # Select a subset if requested
    if args.max_eval_samples and args.max_eval_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(args.max_eval_samples))
        print(f"Using a subset of {len(eval_dataset)} samples for evaluation.")

    print("Preprocessing evaluation dataset...")
    original_columns = eval_dataset.column_names # Keep track for later display
    eval_dataset_processed = eval_dataset.map(
        preprocess_anthropic_hh, batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=original_columns # Remove text columns after processing
    )

    # Filter out examples where chosen/rejected are identical after tokenization
    eval_dataset_processed = eval_dataset_processed.filter(
        lambda x: x['input_ids_chosen'] != x['input_ids_rejected']
    )
    print(f"Filtered eval examples: {len(eval_dataset_processed)}")
    eval_dataset_processed.set_format(type="torch")


    # --- 3. Run Inference and Calculate Accuracy ---
    print("Running inference to get preference scores...")
    total_correct = 0
    total_pairs = 0
    all_scores_chosen = []
    all_scores_rejected = []

    # Manual batching for inference
    data_loader = torch.utils.data.DataLoader(eval_dataset_processed, batch_size=args.batch_size)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Pairs"):
            # Prepare inputs for chosen and rejected, move to device
            inputs_chosen = {
                "input_ids": batch["input_ids_chosen"].to(device),
                "attention_mask": batch["attention_mask_chosen"].to(device),
            }
            inputs_rejected = {
                "input_ids": batch["input_ids_rejected"].to(device),
                "attention_mask": batch["attention_mask_rejected"].to(device),
            }

            # Get scores from the model
            scores_chosen = model(**inputs_chosen).logits.squeeze().cpu() # Shape: (batch_size,)
            scores_rejected = model(**inputs_rejected).logits.squeeze().cpu() # Shape: (batch_size,)

            # Ensure scores are iterable even for batch size 1
            if scores_chosen.dim() == 0: scores_chosen = scores_chosen.unsqueeze(0)
            if scores_rejected.dim() == 0: scores_rejected = scores_rejected.unsqueeze(0)

            # Check accuracy for the batch
            correct_in_batch = (scores_chosen > scores_rejected).sum().item()
            total_correct += correct_in_batch
            total_pairs += len(scores_chosen)

            all_scores_chosen.extend(scores_chosen.numpy())
            all_scores_rejected.extend(scores_rejected.numpy())

            # Clear GPU memory periodically
            if total_pairs % (args.batch_size * 10) == 0:
                 gc.collect()
                 if device == torch.device("cuda"):
                     torch.cuda.empty_cache()

    # Calculate overall accuracy
    accuracy = (total_correct / total_pairs) * 100 if total_pairs > 0 else 0

    print("\n--- Preference Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.2f}% ({total_correct}/{total_pairs} pairs correctly ranked)")
    if all_scores_chosen:
         print(f"   - Average Score (Chosen):   {np.mean(all_scores_chosen):.4f}")
         print(f"   - Average Score (Rejected): {np.mean(all_scores_rejected):.4f}")
         print(f"   - Average Score Difference: {np.mean(all_scores_chosen) - np.mean(all_scores_rejected):.4f}")

    # --- 4. Show Example Scores (Optional) ---
    print("\n--- Example Scores ---")
    num_examples_to_show = 5
    for i in range(min(num_examples_to_show, len(eval_dataset))):
         original_example = eval_dataset[i] # Get original text
         chosen_text = original_example['chosen'][:150] + "..."
         rejected_text = original_example['rejected'][:150] + "..."
         score_chosen = all_scores_chosen[i]
         score_rejected = all_scores_rejected[i]
         is_correct = score_chosen > score_rejected

         print(f"\n--- Example {i+1} ---")
         print(f"  Chosen Text:   \"{chosen_text}\"")
         print(f"  Rejected Text: \"{rejected_text}\"")
         print(f"  Score (Chosen):   {score_chosen:.4f}")
         print(f"  Score (Rejected): {score_rejected:.4f}")
         print(f"  Model Prediction Correct? {'Yes' if is_correct else 'No'}")
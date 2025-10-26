import argparse
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
import gc

# --- Constants ---
CIVIL_TEXT_COL = "text"
TOXICITY_COLUMNS = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']
DATASET_NAME = "civil_comments"

# --- Helper Functions ---
def get_probability(logit):
    """Applies sigmoid function to convert a logit to probability."""
    if isinstance(logit, np.ndarray):
        logit = torch.from_numpy(logit)
    elif not isinstance(logit, torch.Tensor):
        logit = torch.tensor(float(logit)) # Ensure it's float

    # Handle potential NaNs or Infs before sigmoid
    if torch.isnan(logit) or torch.isinf(logit):
        return 0.5 # Return neutral probability for invalid inputs
    try:
        prob = torch.sigmoid(logit).item()
    except Exception:
        prob = 0.5 # Fallback
    return prob

def process_and_tokenize_civil_comments(dataset, tokenizer, max_len):
    """
    Loads civil_comments data, calculates the max toxicity score, and tokenizes.
    Mirrors the training script's data preparation.
    """
    print("Processing and tokenizing civil_comments data...")

    def is_valid(example):
        if example[CIVIL_TEXT_COL] is None: return False
        for col in TOXICITY_COLUMNS:
            if example[col] is None or not isinstance(example[col], (float, int)): # Allow ints just in case
                return False
        return True

    dataset = dataset.filter(is_valid)

    def create_regression_label(example):
        scores = [float(example[col]) for col in TOXICITY_COLUMNS]
        max_score = np.max(scores) if scores else 0.0 # Handle empty scores list
        return {"text": example[CIVIL_TEXT_COL], "label": max_score}

    # Apply mapping and handle potential errors during map
    try:
        # Keep original columns needed for later display if showing examples
        original_cols = dataset.column_names
        dataset = dataset.map(create_regression_label, remove_columns=[col for col in original_cols if col not in [CIVIL_TEXT_COL] + TOXICITY_COLUMNS])
    except Exception as e:
         print(f"Error during label creation map: {e}")
         # Attempt to proceed with potentially fewer columns if error is recoverable
         pass

    def preprocess_function(examples):
        tokenized_batch = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )
        tokenized_batch["label"] = [np.float32(label) for label in examples["label"]]
        return tokenized_batch

    try:
        # Keep 'text' column temporarily if needed for display later
        tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=[col for col in dataset.column_names if col not in ['text', 'label']])
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    except Exception as e:
        print(f"Error during tokenization map: {e}")
        return None # Indicate failure

    print(f"Processed and tokenized {len(tokenized_dataset)} samples.")
    return tokenized_dataset

# --- Main Evaluation Function ---
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    print(f"Loading model and tokenizer from '{args.model_dir}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model/tokenizer: {e}")
        return

    print(f"Loading '{DATASET_NAME}' validation split...")
    try:
        eval_dataset_raw = load_dataset(DATASET_NAME, split="validation")
        # Keep track of original texts if needed for display later
        original_texts_for_display = eval_dataset_raw[CIVIL_TEXT_COL]

        if args.eval_subset_size > 0:
            # Important: Shuffle BEFORE selecting to get a random subset
            eval_dataset_shuffled = eval_dataset_raw.shuffle(seed=42)
            eval_dataset_raw = eval_dataset_shuffled.select(range(args.eval_subset_size))
            # Keep track of the texts corresponding to the selected indices
            original_texts_for_display = eval_dataset_raw[CIVIL_TEXT_COL]
            print(f"Using a shuffled subset of {args.eval_subset_size} samples for evaluation.")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    tokenized_eval_dataset = process_and_tokenize_civil_comments(eval_dataset_raw, tokenizer, args.max_len)
    if tokenized_eval_dataset is None: return # Exit if tokenization failed

    del eval_dataset_raw # Free memory
    gc.collect()

    print("Getting predictions from the model...")
    dummy_args = TrainingArguments(
        output_dir="./temp_eval_classifier",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        fp16=torch.cuda.is_available()
    )
    trainer = Trainer(model=model, args=dummy_args)

    try:
        predictions_output = trainer.predict(tokenized_eval_dataset)
        logits = predictions_output.predictions.flatten()
        labels = predictions_output.label_ids.flatten() # True max_score labels
        if len(logits) != len(labels):
            print(f"Warning: Mismatch between logit count ({len(logits)}) and label count ({len(labels)}). Indices might be misaligned.")
            # Attempt to align if possible, otherwise examples might be wrong
            min_len = min(len(logits), len(labels))
            logits = logits[:min_len]
            labels = labels[:min_len]

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        # Try to clean up memory even on error
        del tokenized_eval_dataset
        gc.collect()
        return

    # Clean up tokenized dataset from memory AFTER successful prediction
    del tokenized_eval_dataset
    gc.collect()

    print("Calculating probabilities from logits...")
    probabilities = np.array([get_probability(logit) for logit in logits])

    # --- 6. Calculate Metrics (Same as before) ---
    print("Calculating evaluation metrics...")
    mse = mean_squared_error(labels, logits)
    try:
        corr = np.corrcoef(labels, probabilities)[0, 1]
    except Exception as e:
        corr = f"N/A ({e})"

    binary_predictions = (probabilities >= 0.5).astype(int)
    binary_labels = (labels >= 0.5).astype(int)
    accuracy = accuracy_score(binary_labels, binary_predictions)
    try:
        if len(np.unique(binary_labels)) > 1:
             roc_auc = roc_auc_score(binary_labels, probabilities)
        else:
             roc_auc = "N/A (Only one class in labels)"
    except Exception as e:
        roc_auc = f"N/A ({e})"

    print("\n--- Evaluation Results ---")
    print(f"📊 Mean Squared Error (MSE) on logits: {mse:.4f}")
    print(f"📈 Correlation (r) between labels and probabilities: {corr if isinstance(corr, str) else f'{corr:.4f}'}")
    print("-" * 25)
    print(f"🎯 Accuracy (at 0.5 probability threshold): {accuracy:.4f} (Note: Can be misleading on imbalanced data)")
    print(f"📈 ROC AUC Score (using probabilities): {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")


    # --- ✨ 7. Show HIGHLY TOXIC Example Scores ---
    print("\n--- Example Predictions (Highly Toxic Comments) ---")
    try:
        # Find indices where the TRUE max_score label is high
        highly_toxic_indices = np.where(labels >= args.toxicity_threshold)[0]

        if len(highly_toxic_indices) == 0:
            print(f"No comments found with True Max Score >= {args.toxicity_threshold} in the evaluated set.")
        else:
            print(f"Showing up to 5 examples with True Max Score >= {args.toxicity_threshold}:")
            # Select a few indices to display (up to 5)
            indices_to_show = highly_toxic_indices[:5]

            for idx in indices_to_show:
                # Get the original text using the index. Make sure alignment is correct.
                # 'original_texts_for_display' should correspond to the order of 'labels' and 'probabilities'
                # if shuffling/subsetting was handled correctly.
                if idx < len(original_texts_for_display):
                    text = original_texts_for_display[idx][:200] + "..."
                else:
                    text = f"[Text unavailable - index {idx} out of bounds]"

                true_label_score = float(labels[idx])
                pred_prob = float(probabilities[idx])
                print(f"\nText: \"{text}\"")
                print(f"  True Max Score: {true_label_score:.4f}")
                print(f"  Predicted Probability (Undesirable): {pred_prob:.4f}")

    except Exception as e:
        print(f"Could not display highly toxic examples: {e}")
        import traceback
        traceback.print_exc() # Print detailed error


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the toxicity/undesirability classifier, focusing on highly toxic examples.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the saved classifier model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length used during training.")
    parser.add_argument("--eval_subset_size", type=int, default=1000, help="Number of validation samples to use (0 for all). Default: 1000 for speed.")
    parser.add_argument("--toxicity_threshold", type=float, default=0.8, help="Threshold for 'highly toxic' (True Max Score).") # Added argument
    args = parser.parse_args()

    main(args)
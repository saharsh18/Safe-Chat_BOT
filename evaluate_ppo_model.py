import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import jinja2

from reward_utils import set_seed, get_raw_preference_scores, get_toxicity_scores

SPECIAL_TOKENS = {"user": "<|user|>", "bot": "<|bot|>", "end": "<|endofbot|>"}
PREF_MEAN = 0.3904
PREF_STD = 1.8893
TOX_MEAN = 0.4729
TOX_STD = 0.0424
EPSILON = 1e-8
W_PREFERENCE = 1
W_TOXICITY = PREF_STD / (TOX_STD + EPSILON)  
INVALID_RESPONSE_REWARD = -2.0



def format_prompt(text, tokenizer):
    """Formats a raw text prompt into the model's chat template, with a fallback."""
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (ValueError, jinja2.exceptions.TemplateError):
        print("Tokenizer has no chat template, using manual format: <|user|>...<|bot|>")
        return f"{SPECIAL_TOKENS['user']}{text}{SPECIAL_TOKENS['bot']}"


def get_combined_rewards(prompts_text, responses_text, pref_model, pref_tokenizer, tox_model, tox_tokenizer, device, args):
    """
    Calculates combined reward and returns raw scores for analysis.
    Returns:
        tuple: (final_scores, raw_pref_scores, raw_safety_scores)
    """
    if not prompts_text or not responses_text:
        return [], [], []

    # 1. Get raw and normalized preference scores
    pref_inputs = [p + r for p, r in zip(prompts_text, responses_text)]
    raw_pref_scores = get_raw_preference_scores(pref_model, pref_tokenizer, pref_inputs, args.reward_batch_size, device, args.pref_max_len)
    raw_pref_scores = np.array(raw_pref_scores)
    norm_pref_scores = (raw_pref_scores - PREF_MEAN) 

    # 2. Get raw and normalized toxicity scores
    raw_safety_scores = get_toxicity_scores(tox_model, tox_tokenizer, responses_text, args.reward_batch_size, device, args.tox_max_len)
    raw_safety_scores = np.array(raw_safety_scores)
    norm_safety_scores = (raw_safety_scores - TOX_MEAN) 

    # 3. Combine NORMALIZED scores for the final reward
    final_scores = (W_PREFERENCE * norm_pref_scores) + (W_TOXICITY * norm_safety_scores)

    return final_scores.tolist(), norm_pref_scores.tolist(), norm_safety_scores.tolist()


def safe_to_string(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)

def process_daily_dialog_prompts(tokenizer, limit, cache_dir):
    print(f"Loading DailyDialog test prompts (limit {limit})...")
    try:
        # Use the 'test' split for evaluation
        raw = load_dataset("daily_dialog", split="test", cache_dir=cache_dir)
        if limit and len(raw) > limit:
            raw = raw.shuffle(seed=42).select(range(limit))
    except Exception as e:
        print(f"Could not load DailyDialog: {e}. Skipping.")
        return None
    
    prompts = [{"prompt": format_prompt(safe_to_string(ex["dialog"][0]), tokenizer)} for ex in raw]
    print(f"Loaded {len(prompts)} prompts from DailyDialog.")
    return Dataset.from_list(prompts) if prompts else None

def process_oasst1_prompts(tokenizer, limit, cache_dir):
    print(f"Loading OASST1 validation prompts (limit {limit})...")
    try:
        # Use the 'validation' split as a test set
        ds = load_dataset("OpenAssistant/oasst1", split="validation", cache_dir=cache_dir)
    except Exception as e:
        print(f"Could not load OASST1: {e}. Skipping.")
        return None
    
    ds_filtered = ds.filter(lambda x: x.get("lang", "") == "en" and x.get("parent_id") is None and x.get("role") == "prompter")
    if limit and len(ds_filtered) > limit:
        ds_filtered = ds_filtered.shuffle(seed=42).select(range(limit))
        
    prompts = [{"prompt": format_prompt(safe_to_string(ex["text"]), tokenizer)} for ex in ds_filtered]
    print(f"Loaded {len(prompts)} prompts from OASST1.")
    return Dataset.from_list(prompts) if prompts else None

def process_anthropic_hh_prompts(tokenizer, limit, cache_dir):
    print(f"Loading Anthropic HH test prompts (limit {limit})...")
    prompts = []
    loaded = 0
    try:
        for subset in ["harmless-base", "helpful-base"]:
            if loaded >= limit: break
            # Use the 'test' split for evaluation
            ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="test", cache_dir=cache_dir,download_mode="force_redownload")
            ds = ds.shuffle(seed=42)
            for ex in ds:
                if loaded >= limit: break
                conversation = ex.get("chosen", "")
                human_turns = conversation.split("\n\nHuman:")
                if len(human_turns) > 1:
                    first_prompt_text = human_turns[1].split("\n\nAssistant:")[0].strip()
                    if first_prompt_text:
                        prompts.append({"prompt": format_prompt(safe_to_string(first_prompt_text), tokenizer)})
                        loaded += 1
    except Exception as e:
        print(f"Could not load Anthropic/hh-rlhf: {e}")
    print(f"Loaded {len(prompts)} prompts from Anthropic HH.")
    return Dataset.from_list(prompts) if prompts else None

def process_real_toxicity_prompts(tokenizer, limit, cache_dir):
    print(f"Loading RealToxicityPrompts (limit {limit})...")
    try:
        # This dataset only has a 'train' split, so we take a sample from it for evaluation
        ds = load_dataset("allenai/real-toxicity-prompts", split="train", cache_dir=cache_dir)
        # Shuffling with a different seed to get a different sample than training
        ds = ds.shuffle(seed=123).select(range(min(limit, len(ds))))
    except Exception as e:
        print(f"Could not load RealToxicityPrompts: {e}. Skipping.")
        return None
    
    prompts = [{"prompt": format_prompt(safe_to_string(ex["prompt"]["text"]), tokenizer)} for ex in ds]
    print(f"Loaded {len(prompts)} prompts from RealToxicityPrompts.")
    return Dataset.from_list(prompts) if prompts else None

def get_evaluation_dataset(tokenizer, args):
    """
    Loads and combines the test/validation splits of the datasets used in training.
    """
    print("--- Loading and Combining Evaluation Datasets ---")
    # Define limits for each dataset for evaluation
    limits = {
        "daily_dialog": 1500,
        "oasst1": 1000,
        "hh": 2000,
        "toxicity": 2500
    }

    datasets_to_combine = []
    
    # Daily Dialog
    d = process_daily_dialog_prompts(tokenizer, limits["daily_dialog"], args.cache_dir)
    if d: datasets_to_combine.append(d)
    
    # OASST1
    d = process_oasst1_prompts(tokenizer, limits["oasst1"], args.cache_dir)
    if d: datasets_to_combine.append(d)
    
    # Anthropic HH
    d = process_anthropic_hh_prompts(tokenizer, limits["hh"], args.cache_dir)
    if d: datasets_to_combine.append(d)
    
    # Real Toxicity Prompts
    d = process_real_toxicity_prompts(tokenizer, limits["toxicity"], args.cache_dir)
    if d: datasets_to_combine.append(d)

    if not datasets_to_combine:
        raise ValueError("No evaluation data loaded!")

    # Combine and shuffle all loaded datasets
    from datasets import concatenate_datasets
    combined = concatenate_datasets(datasets_to_combine).shuffle(seed=42)
    
    # Apply the global dataset limit if one was passed
    if args.dataset_limit and len(combined) > args.dataset_limit:
        combined = combined.select(range(args.dataset_limit))
        
    print(f"Total combined prompts for evaluation: {len(combined)}")
    print("-------------------------------------------------")
    return combined

def main(args):
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this evaluation script.")
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # --- Load Models and Tokenizers ---
    print("Loading models...")
    # PPO Model to evaluate
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, cache_dir=args.cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, cache_dir=args.cache_dir, padding_side="left")
    
    if tokenizer.pad_token is None:
        print("Tokenizer has no pad token; setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id

    special_tokens_dict = {'additional_special_tokens': list(SPECIAL_TOKENS.values())}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    end_token = SPECIAL_TOKENS["end"]
    end_token_id = tokenizer.convert_tokens_to_ids(end_token)
    pad_token_id = tokenizer.pad_token_id
    user_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["user"])

    
    reward_device = torch.device(args.reward_device)
    tox_model = AutoModelForSequenceClassification.from_pretrained(args.tox_model_dir, cache_dir=args.cache_dir).to(reward_device)
    pref_model = AutoModelForSequenceClassification.from_pretrained(args.pref_model_dir, cache_dir=args.cache_dir).to(reward_device)
    tox_tokenizer = AutoTokenizer.from_pretrained(args.tox_model_dir, cache_dir=args.cache_dir)
    pref_tokenizer = AutoTokenizer.from_pretrained(args.pref_model_dir, cache_dir=args.cache_dir)
    print("Models loaded.")

    # --- Load Dataset ---
    eval_dataset = get_evaluation_dataset(tokenizer, args)
    if eval_dataset is None:
        print("Failed to load dataset. Exiting.")
        return

    # --- Generation Settings ---
    generation_kwargs = {
        "min_length": -1,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 1.0,
        "pad_token_id":pad_token_id,
        "eos_token_id":[end_token_id, user_token_id],
        "max_new_tokens": args.output_max_len,
    }

    # --- Evaluation Loop ---
    all_rewards = []
    all_pref_scores = []
    all_tox_scores = []
    pbar = tqdm(range(0, len(eval_dataset), args.batch_size), desc="Evaluating Batches")

    for i in pbar:
        batch_prompts = eval_dataset[i:i+args.batch_size]["prompt"]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.input_max_len).to(device)
        
        # Generate responses
        response_tensors = model.generate(**inputs, **generation_kwargs)
        
        # Decode prompts and responses
        response_only_tensors = response_tensors[:, inputs['input_ids'].shape[1]:]
        response_texts = tokenizer.batch_decode(response_only_tensors, skip_special_tokens=True)

        # Calculate rewards
        valid_indices = [j for j, r in enumerate(response_texts) if r and r.strip()]
        
        if not valid_indices:
            batch_rewards = [INVALID_RESPONSE_REWARD] * len(batch_prompts)
        else:
            valid_prompts = [batch_prompts[j] for j in valid_indices]
            valid_responses = [response_texts[j] for j in valid_indices]
            
            combined_scores, raw_pref, raw_tox = get_combined_rewards(
                valid_prompts, valid_responses,
                pref_model, pref_tokenizer, tox_model, tox_tokenizer,
                reward_device, args
            )
            
            # Store raw scores for analysis
            all_pref_scores.extend(raw_pref)
            all_tox_scores.extend(raw_tox)
            
            batch_rewards = []
            score_idx = 0
            for j in range(len(batch_prompts)):
                if j in valid_indices:
                    batch_rewards.append(combined_scores[score_idx])
                    score_idx += 1
                else:
                    batch_rewards.append(INVALID_RESPONSE_REWARD)

        all_rewards.extend(batch_rewards)
        pbar.set_postfix({"mean_reward": f"{np.mean(all_rewards):.4f}"})

    # --- Final Results ---
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_pref_score = np.mean(all_pref_scores) if all_pref_scores else 0
    mean_tox_score = np.mean(all_tox_scores) if all_tox_scores else 0
    num_samples = len(all_rewards)

    print("\n--- Evaluation Complete ---")
    print(f"Model: {args.model_dir}")
    print(f"Evaluated on {num_samples} samples.")
    print(f"Average Reward: {mean_reward:.4f}")
    print(f"Standard Deviation of Reward: {std_reward:.4f}")
    print(f"Mean Preference Score (raw): {mean_pref_score:.4f}")
    print(f"Mean Toxicity Score (raw): {mean_tox_score:.4f}")
    print("---------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PPO-trained model.")
    # Model paths
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained PPO model directory.")
    parser.add_argument("--tox_model_dir", type=str, required=True, help="Path to the toxicity classifier model.")
    parser.add_argument("--pref_model_dir", type=str, required=True, help="Path to the preference model.")
    
    # Directory and device settings
    parser.add_argument("--output_dir", type=str, default="temp_eval_results", help="Directory to save any potential outputs.")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory.")
    parser.add_argument("--reward_device", type=str, default="cuda", help="Device for reward models ('cuda' or 'cpu').")

    # Length and batching settings
    parser.add_argument("--input_max_len", type=int, default=128, help="Max length for input prompts.")
    parser.add_argument("--output_max_len", type=int, default=64, help="Max length for generated responses.")
    parser.add_argument("--tox_max_len", type=int, default=128, help="Max length for toxicity model input.")
    parser.add_argument("--pref_max_len", type=int, default=512, help="Max length for preference model input.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--reward_batch_size", type=int, default=16, help="Batch size for reward calculation.")
    
    # Dataset and execution settings
    parser.add_argument("--dataset_limit", type=int, default=8000, help="Number of samples to use from the evaluation dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)

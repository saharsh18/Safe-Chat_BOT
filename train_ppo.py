import argparse
import torch
import gc
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLMWithValueHead, # Use the ValueHead model for PPO Actor
    TrainingArguments # Used for dummy args
)
from trl import PPOTrainer, PPOConfig, create_reference_model
from collections import defaultdict
import random
from tqdm import tqdm
import os
import time

# --- Constants ---
# Special tokens MUST match the ones used for the baseline SFT model
SPECIAL_TOKENS = {
    "user": "<|user|>",
    "bot": "<|bot|>",
    "end": "<|endofbot|>",
}

# --- Prompt Dataset Configuration ---
MAX_DAILYDIALOG_PROMPTS = 7000 # Benign
MAX_OASST1_PROMPTS = 3000      # Benign
MAX_HH_PROMPTS = 2000          # Provocative
MAX_TOXICITY_PROMPTS = 2000    # Provocative

# --- Reward Weights ---
# These are crucial and require tuning!
W_PREFERENCE = 1.0 # Weight for the preference model score
W_TOXICITY = 1.0   # Weight for the (1 - toxicity_prob) score

# --- Data Processing Functions ---

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_prompt(text):
    """Applies the SFT chat template to a single user turn."""
    messages = [{"role": "user", "content": text}]
    # apply_chat_template will add the <|user|> and <|bot|> tokens
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def process_daily_dialog_prompts():
    """Loads and formats DailyDialog user prompts (Benign)."""
    print(f"Loading DailyDialog prompts (limit {MAX_DAILYDIALOG_PROMPTS})...")
    try:
        raw = load_dataset("daily_dialog", split="train", cache_dir=args.cache_dir)
    except Exception as e:
        print(f"Could not load DailyDialog: {e}. Skipping.")
        return None
    prompts = []
    processed_dialog_ids = set()
    raw = raw.shuffle(seed=42)
    for example in raw:
        if len(prompts) >= MAX_DAILYDIALOG_PROMPTS: break
        dialog_id = example.get('id', hash(tuple(example['dialog'])))
        if dialog_id in processed_dialog_ids: continue
        user_turns = example["dialog"][::2] # Get user utterances
        if user_turns:
             prompts.append({"prompt": format_prompt(user_turns[0])}) # Use first user turn
             processed_dialog_ids.add(dialog_id)
    print(f"Loaded {len(prompts)} prompts from DailyDialog.")
    return Dataset.from_list(prompts) if prompts else None


def process_oasst1_prompts():
    """Loads and formats OASST1 initial user prompts (Benign)."""
    print(f"Loading OASST1 prompts (limit {MAX_OASST1_PROMPTS})...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train", cache_dir=args.cache_dir)
    except Exception as e:
        print(f"Could not load OASST1: {e}. Skipping.")
        return None
    ds_filtered = ds.filter(lambda x: x["lang"] == "en" and x["parent_id"] is None and x["role"] == "prompter")
    if len(ds_filtered) > MAX_OASST1_PROMPTS:
         ds_filtered = ds_filtered.shuffle(seed=42).select(range(MAX_OASST1_PROMPTS))
    prompts = [{"prompt": format_prompt(example["text"])} for example in ds_filtered]
    print(f"Loaded {len(prompts)} prompts from OASST1.")
    return Dataset.from_list(prompts) if prompts else None

def process_anthropic_hh_prompts():
    """Loads human prompts from Anthropic HH (Provocative)."""
    print(f"Loading Anthropic HH prompts (limit {MAX_HH_PROMPTS})...")
    try:
        # Load both helpful and harmless subsets to get a wide range of prompts
        ds_harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train", cache_dir=args.cache_dir)
        ds_helpful = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train", cache_dir=args.cache_dir)
        ds = concatenate_datasets([ds_harmless, ds_helpful]).shuffle(seed=42)
    except Exception as e:
        print(f"Could not load Anthropic/hh-rlhf: {e}. Skipping.")
        return None

    prompts = []
    # Extract the first human turn from the conversation
    for example in ds:
        if len(prompts) >= MAX_HH_PROMPTS: break
        # Conversation is in 'chosen' (and 'rejected') field
        conversation = example['chosen']
        # Find the first occurrence of "\n\nHuman:"
        human_turns = conversation.split("\n\nHuman:")
        if len(human_turns) > 1:
            # Get the text after the first "\n\nHuman:" and before "\n\nAssistant:"
            first_prompt = human_turns[1].split("\n\nAssistant:")[0].strip()
            if first_prompt:
                prompts.append({"prompt": format_prompt(first_prompt)})
                
    print(f"Loaded {len(prompts)} prompts from Anthropic HH.")
    return Dataset.from_list(prompts) if prompts else None

def process_real_toxicity_prompts():
    """Loads prompts from RealToxicityPrompts (Provocative)."""
    print(f"Loading RealToxicityPrompts (limit {MAX_TOXICITY_PROMPTS})...")
    try:
        ds = load_dataset("allenai/real-toxicity-prompts", split="train", cache_dir=args.cache_dir)
        ds = ds.shuffle(seed=42).select(range(MAX_TOXICITY_PROMPTS))
    except Exception as e:
        print(f"Could not load allenai/real-toxicity-prompts: {e}. Skipping.")
        return None
    
    # These are text fragments, we treat them as the user's prompt
    prompts = [{"prompt": format_prompt(example["prompt"]["text"])} for example in ds]
    print(f"Loaded {len(prompts)} prompts from RealToxicityPrompts.")
    return Dataset.from_list(prompts) if prompts else None


def get_prompt_dataset(tokenizer_):
    """Combines prompts from benign and provocative sources."""
    global tokenizer # Make tokenizer global for data processing
    tokenizer = tokenizer_

    # Load all sources
    d_dialog = process_daily_dialog_prompts()
    d_oasst1 = process_oasst1_prompts()
    d_hh = process_anthropic_hh_prompts()
    d_toxic = process_real_toxicity_prompts()

    # Combine benign and provocative
    datasets_to_combine = [d for d in [d_dialog, d_oasst1, d_hh, d_toxic] if d is not None]
    if not datasets_to_combine:
         raise ValueError("No prompt data could be loaded!")

    combined = concatenate_datasets(datasets_to_combine).shuffle(seed=42)

    def tokenize_prompt(element):
        tokenized = tokenizer_(element["prompt"], truncation=True, max_length=args.input_max_len)
        return {"input_ids": tokenized["input_ids"], "query": element["prompt"]}

    combined = combined.map(tokenize_prompt, remove_columns=["prompt"])
    combined.set_format(type="torch")
    print(f"Total prompts for PPO: {len(combined)}")
    return combined

# --- Reward Calculation Function ---

def get_combined_rewards(
    prompts_text, responses_text,
    pref_model, pref_tokenizer,
    tox_model, tox_tokenizer,
    device, args
):
    """
    Calculates the combined reward signal using Preference and Toxicity models.
    Returns a tensor of final scores, ready for PPO.
    """
    batch_size = len(prompts_text)
    pref_scores_list = [0.0] * batch_size
    tox_scores_list = [0.0] * batch_size

    # --- 1. Preference Model Score ---
    # The PM scores the (prompt + response) concatenation
    with torch.no_grad():
        # Important: The PM was trained on Anthropic format (e.g., "\n\nHuman: ... \n\nAssistant: ...")
        # Our `format_prompt` creates <|user|>...<|bot|>. This mismatch is OK
        # as long as the PM is a general model like DistilBERT/RoBERTa,
        # but a PM trained on <|user|> format would be ideal.
        # For now, we simply concatenate the raw text.
        
        # We must decode the prompt *text* from the tensors, but batch["query"] is already text
        # The PM needs the *full conversation context* + response.
        # Our `prompts_text` already includes the full context up to <|bot|>
        inputs_for_pref = pref_tokenizer(
            [p + r for p, r in zip(prompts_text, responses_text)], # Concat prompt + response
            padding=True,
            truncation=True,
            max_length=args.pref_max_len,
            return_tensors="pt"
        ).to(device)

        pref_outputs = pref_model(**inputs_for_pref)
        pref_scores_raw = pref_outputs.logits.squeeze() # Shape: (batch_size,)
        # Handle single-item batch
        if pref_scores_raw.dim() == 0: pref_scores_raw = pref_scores_raw.unsqueeze(0)
        pref_scores_list = pref_scores_raw.cpu().tolist()

    # --- 2. Toxicity Model Score ---
    # The Toxicity model scores *only the response*
    with torch.no_grad():
        inputs_for_tox = tox_tokenizer(
            responses_text, # Score only the response text
            padding=True,
            truncation=True,
            max_length=args.tox_max_len,
            return_tensors="pt"
        ).to(device)

        tox_outputs = tox_model(**inputs_for_tox)
        tox_logits = tox_outputs.logits.squeeze() # Shape: (batch_size,)
        # Handle single-item batch
        if tox_logits.dim() == 0: tox_logits = tox_logits.unsqueeze(0)
        
        probs_undesirable = torch.sigmoid(tox_logits)
        probs_undesirable_list = probs_undesirable.cpu().tolist()
        # Higher score = safer (1 - prob_undesirable)
        tox_scores_list = [1.0 - p for p in probs_undesirable_list]

    # --- 3. Combine Rewards ---
    final_scores = []
    for i in range(batch_size):
        # Weighted sum
        score = (
            W_PREFERENCE * pref_scores_list[i] +
            W_TOXICITY * tox_scores_list[i]
        )
        final_scores.append(score)

    # Return as tensor on the correct device
    return torch.tensor(final_scores, dtype=torch.float32, device=device)


# --- Main PPO Training ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 Medium with PPO using combined rewards.")
    # Model Paths
    parser.add_argument("--sft_model_dir", type=str, required=True, help="Path to the fine-tuned baseline SFT model.")
    parser.add_argument("--tox_model_dir", type=str, required=True, help="Path to the trained toxicity classifier model.")
    parser.add_argument("--pref_model_dir", type=str, required=True, help="Path to the trained preference model.")
    parser.add_argument("--output_dir", type=str, default="gpt2-medium-ppo-combined", help="Directory to save the PPO-trained model.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face cache directory.")

    # Dataset/Tokenization Params
    parser.add_argument("--input_max_len", type=int, default=128, help="Max length for input prompts during PPO.")
    parser.add_argument("--output_max_len", type=int, default=128, help="Max length for generated responses during PPO.")
    parser.add_argument("--tox_max_len", type=int, default=128, help="Max length the toxicity classifier was trained on.")
    parser.add_argument("--pref_max_len", type=int, default=512, help="Max length the preference model was trained on.")

    # PPO Hyperparameters
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO optimization epochs per batch.")
    parser.add_argument("--learning_rate", type=float, default=1.41e-6, help="Adam learning rate for PPO.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generating responses (adjust VRAM).")
    parser.add_argument("--mini_batch_size", type=int, default=4, help="Mini-batch size for PPO updates (adjust VRAM).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation.")
    parser.add_argument("--adap_kl_ctrl", type=bool, default=True, help="Use adaptive KL control.")
    parser.add_argument("--init_kl_coef", type=float, default=0.05, help="Initial KL coefficient.")
    parser.add_argument("--target_kl", type=float, default=0.05, help="Target KL value (lower = less deviation).")
    parser.add_argument("--clip_reward", type=float, default=10.0, help="Clip rewards magnitude.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # --- GPU Check ---
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. PPO training requires a GPU.")
        exit()
    else:
        device = torch.device("cuda")
        print(f"CUDA available. Using {torch.cuda.get_device_name(0)}")

    # --- 1. Load Models and Tokenizers ---
    print("Loading models and tokenizers...")
    torch_dtype = torch.bfloat16 # Use bfloat16 for all models

    # SFT/PPO Tokenizer (used for generation)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set SFT tokenizer pad_token to eos_token.")

    # Toxicity Classifier Tokenizer
    tox_tokenizer = AutoTokenizer.from_pretrained(args.tox_model_dir, cache_dir=args.cache_dir)

    # Preference Model Tokenizer
    pref_tokenizer = AutoTokenizer.from_pretrained(args.pref_model_dir, cache_dir=args.cache_dir)

    # PPO Actor Model (with Value Head)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.sft_model_dir,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir
    ).to(device)

    # Reference Model (frozen copy of the SFT model)
    ref_model = create_reference_model(model).to(device)
    ref_model.eval()

    # Toxicity Classifier Model
    tox_model = AutoModelForSequenceClassification.from_pretrained(
        args.tox_model_dir,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir
    ).to(device)
    tox_model.eval()

    # Preference Model
    pref_model = AutoModelForSequenceClassification.from_pretrained(
        args.pref_model_dir,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir
    ).to(device)
    pref_model.eval()

    print("Models and tokenizers loaded.")

    # --- 2. Prepare Prompt Dataset ---
    print("Preparing prompt dataset...")
    dataset = get_prompt_dataset(tokenizer)

    # --- 3. Configure PPO ---
    ppo_config = PPOConfig(
        model_name=args.sft_model_dir,
        learning_rate=args.learning_rate,
        log_with="tensorboard",
        project_kwargs={"logging_dir": os.path.join(args.output_dir, "logs")},
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        kl_penalty="kl",
        target=args.target_kl,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=args.adap_kl_ctrl,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        use_score_norm = True, # Normalize combined rewards
        score_clip = args.clip_reward, # Clip normalized rewards
    )

    # --- 4. Initialize PPOTrainer ---
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    # --- 5. Define Generation Kwargs ---
    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end"]),
        "max_new_tokens": args.output_max_len,
    }

    # --- 6. PPO Training Loop ---
    print("Starting PPO training...")
    total_ppo_steps = ppo_config.total_ppo_epochs * len(ppo_trainer.dataloader)

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=total_ppo_steps, desc="PPO Steps"):
        query_tensors = batch["input_ids"] # Already tokenized prompts
        query_texts = batch["query"]      # Raw prompt text (from get_prompt_dataset)
        
        # Ensure query tensors are on the correct device
        query_tensors = query_tensors.to(device)

        start_time = time.time() # Time generation

        # Generate responses from the Actor model
        with torch.cuda.amp.autocast(dtype=torch_dtype):
             response_tensors = ppo_trainer.generate(
                 query_tensors,
                 return_prompt=False,
                 **generation_kwargs,
             )
        gen_time = time.time() - start_time
        
        # Decode responses to text
        try:
             response_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        except Exception as e:
             print(f"Error decoding responses at step {step}: {e}")
             continue # Skip this batch

        reward_start_time = time.time()

        # Filter out empty responses *before* scoring
        valid_indices = [i for i, r in enumerate(response_texts) if r.strip()]
        
        if not valid_indices:
             print(f"Warning: Step {step}: All generated responses were empty. Assigning default low reward.")
             rewards = torch.full((len(query_tensors),), -10.0, dtype=torch.float32, device=device) # Assign large penalty
        else:
             # Select only valid prompts and responses for scoring
             valid_query_texts = [query_texts[i] for i in valid_indices]
             valid_response_texts = [response_texts[i] for i in valid_indices]

             # Calculate combined reward scores
             combined_scores = get_combined_rewards(
                 valid_query_texts, valid_response_texts,
                 pref_model, pref_tokenizer,
                 tox_model, tox_tokenizer,
                 device, args
             )
             
             # Create a full rewards tensor, assigning a large penalty to empty responses
             rewards = torch.full((len(query_tensors),), -10.0, dtype=torch.float32, device=device)
             for i, score in zip(valid_indices, combined_scores):
                 rewards[i] = score

        reward_time = time.time() - reward_start_time

        # Run PPO step
        # TRL expects lists of tensors
        query_tensors_list = [tensor for tensor in query_tensors]
        response_tensors_list = [tensor for tensor in response_tensors]
        
        try:
            stats = ppo_trainer.step(query_tensors_list, response_tensors_list, rewards)
            # Log custom stats
            stats["custom/gen_time"] = gen_time
            stats["custom/reward_time"] = reward_time
            stats["custom/non_empty_responses"] = len(valid_indices) / len(query_tensors)
            ppo_trainer.log_stats(stats, batch, rewards) 
        except Exception as e:
             print(f"Error during PPO step {step}: {e}")
             # Decide how to handle: skip step, log error, etc.
             continue # Skip this step

        # Optional: Clean GPU memory periodically
        if (step + 1) % 50 == 0:
             gc.collect()
             torch.cuda.empty_cache()

    print("PPO Training complete.")

    # --- 7. Save Final Model ---
    print(f"Saving final PPO model to '{args.output_dir}'...")
    ppo_trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("PPO model saved successfully.")
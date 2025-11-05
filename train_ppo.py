import argparse
import os
import time
import gc
import random
from collections import defaultdict

import torch
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm

from datasets import load_dataset, Dataset, concatenate_datasets

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)

from trl import (
    PPOTrainer,
    PPOConfig,
    create_reference_model,
    AutoModelForCausalLMWithValueHead,
)

from reward_utils import set_seed, get_raw_preference_scores, get_toxicity_scores

# ---------------- CONSTANTS ----------------
SPECIAL_TOKENS = {"user": "<|user|>", "bot": "<|bot|>", "end": "<|endofbot|>"}
PREF_MEAN = 0.4913
PREF_STD = 1.9863
TOX_MEAN = 0.4431
TOX_STD = 0.0878

EPSILON = 1e-8 # For safe division

MAX_DAILYDIALOG_PROMPTS = 10000
MAX_OASST1_PROMPTS = 3000
MAX_HH_PROMPTS = 5000
MAX_TOXICITY_PROMPTS = 2500

W_PREFERENCE = 1.0
W_TOXICITY = PREF_STD / (TOX_STD + EPSILON)   

INVALID_RESPONSE_REWARD = -2.0

# ---------------- Helper Functions ----------------
def format_prompt(text, tokenizer_to_use):
    messages = [{"role": "user", "content": text}]
    return tokenizer_to_use.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def safe_to_string(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    elif isinstance(x, torch.Tensor):
        try:
             return str(x.tolist()) 
        except:
             return str(x)
    elif isinstance(x, (list, dict)):
        return str(x)
    else:
        return str(x)


def process_daily_dialog_prompts(tokenizer_, limit=MAX_DAILYDIALOG_PROMPTS):
    print(f"Loading DailyDialog prompts (limit {limit})...")
    try:
        raw = load_dataset("daily_dialog", split="train", cache_dir=args.cache_dir)
    except Exception as e:
        print(f"Could not load DailyDialog: {e}. Skipping.")
        return None
    prompts = []
    processed_ids = set()
    raw = raw.shuffle(seed=42)
    for ex in raw:
        if len(prompts) >= limit: break
        dialog_id = ex.get("id", hash(tuple(ex["dialog"])))
        if dialog_id in processed_ids: continue
        user_turns = ex["dialog"][::2]
        if user_turns:
            text = safe_to_string(user_turns[0])
            prompts.append({"prompt": format_prompt(text, tokenizer_)})
            processed_ids.add(dialog_id)
    print(f"Loaded {len(prompts)} prompts from DailyDialog.")
    return Dataset.from_list(prompts) if prompts else None

def process_oasst1_prompts(tokenizer_, limit=MAX_OASST1_PROMPTS):
    print(f"Loading OASST1 prompts (limit {limit})...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train", cache_dir=args.cache_dir)
    except Exception as e:
        print(f"Could not load OASST1: {e}. Skipping.")
        return None
    ds_filtered = ds.filter(lambda x: x.get("lang", "") == "en" and x.get("parent_id") is None and x.get("role") == "prompter")
    if len(ds_filtered) > limit:
        ds_filtered = ds_filtered.shuffle(seed=42).select(range(limit))
    prompts = [{"prompt": format_prompt(safe_to_string(example["text"]), tokenizer_)} for example in ds_filtered] # Use safe_to_string
    print(f"Loaded {len(prompts)} prompts from OASST1.")
    return Dataset.from_list(prompts) if prompts else None

def process_anthropic_hh_prompts(tokenizer_, limit=MAX_HH_PROMPTS):
    print(f"Loading Anthropic HH prompts (limit {limit})...")
    prompts = []
    loaded = 0
    try:
        for subset in ["harmless-base", "helpful-base"]:
            if loaded >= limit: break
            ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="train", cache_dir=args.cache_dir)
            ds = ds.shuffle(seed=42)
            for ex in ds:
                if loaded >= limit: break
                conversation = ex.get("chosen", "")
                human_turns = conversation.split("\n\nHuman:")
                if len(human_turns) > 1:
                    first_prompt_text = human_turns[1].split("\n\nAssistant:")[0].strip()
                    if first_prompt_text:
                        prompts.append({"prompt": format_prompt(safe_to_string(first_prompt_text), tokenizer_)}) # Use safe_to_string
                        loaded += 1
    except Exception as e:
        print(f"Could not load Anthropic/hh-rlhf: {e}")
    print(f"Loaded {len(prompts)} prompts from Anthropic HH.")
    return Dataset.from_list(prompts) if prompts else None

def process_real_toxicity_prompts(tokenizer_, limit=MAX_TOXICITY_PROMPTS):
    print(f"Loading RealToxicityPrompts (limit {limit})...")
    try:
        ds = load_dataset("allenai/real-toxicity-prompts", split="train", cache_dir=args.cache_dir)
        ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
    except Exception as e:
        print(f"Could not load RealToxicityPrompts: {e}. Skipping.")
        return None
    prompts = [{"prompt": format_prompt(safe_to_string(example["prompt"]["text"]), tokenizer_)} for example in ds] # Use safe_to_string
    print(f"Loaded {len(prompts)} prompts from RealToxicityPrompts.")
    return Dataset.from_list(prompts) if prompts else None

def get_prompt_dataset(tokenizer_):
    datasets_to_combine = []
    for fn in [process_daily_dialog_prompts,
        process_oasst1_prompts,
        process_anthropic_hh_prompts,
        process_real_toxicity_prompts]:
        d = fn(tokenizer_)
        if d is not None:
            datasets_to_combine.append(d)
    if not datasets_to_combine:
        raise ValueError("No prompt data loaded!")

    combined = concatenate_datasets(datasets_to_combine).shuffle(seed=42)
    print(f"Total combined prompts before tokenization: {len(combined)}")

    def tokenize_prompt(element):
        text = safe_to_string(element["prompt"]) 
        tokenized = tokenizer_(
            text,
            truncation=True,
            max_length=args.input_max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "query": text,
        }

    combined = combined.map(tokenize_prompt, remove_columns=["prompt"], batched=False) # Use batched=False for simplicity

    combined.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print(f"Total prompts for PPO: {len(combined)}")
    return combined

def get_combined_rewards(
    prompts_text, responses_text,
    pref_model, pref_tokenizer,
    tox_model, tox_tokenizer,
    device, args
):
    """Calculates combined, NORMALIZED reward. Returns list of floats."""
    if not prompts_text or not responses_text:
        return []

    # 1. Normalized Preference Score
    pref_inputs = [p + r for p, r in zip(prompts_text, responses_text)]
    raw_pref_scores = get_raw_preference_scores(
        pref_model, pref_tokenizer, pref_inputs,
        args.reward_batch_size, device, args.pref_max_len
    )
    raw_pref_scores = np.array(raw_pref_scores)
    norm_pref_scores = (raw_pref_scores - PREF_MEAN)

    # 2. Normalized Toxicity (Safety) Score
    raw_safety_scores = get_toxicity_scores(
        tox_model, tox_tokenizer, responses_text,
        args.reward_batch_size, device, args.tox_max_len
    )
    raw_safety_scores = np.array(raw_safety_scores)
    norm_safety_scores = (raw_safety_scores - TOX_MEAN)

    # 3. Combine NORMALIZED scores
    final_scores = []
    for pref_score, safety_score in zip(norm_pref_scores, norm_safety_scores):
        score = (W_PREFERENCE * float(pref_score)) + (W_TOXICITY * float(safety_score))
        final_scores.append(float(score)) # Ensure float
        

    return final_scores


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model_dir", type=str, required=True)
    parser.add_argument("--tox_model_dir", type=str, required=True)
    parser.add_argument("--pref_model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="ppo-output-stable-tox")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--input_max_len", type=int, default=128)
    parser.add_argument("--output_max_len", type=int, default=64)
    parser.add_argument("--tox_max_len", type=int, default=128)
    parser.add_argument("--pref_max_len", type=int, default=512)
    parser.add_argument("--reward_batch_size", type=int, default=16)
    parser.add_argument("--ppo_epochs", type=int, default=1) # Fewer epochs can be more stable
    parser.add_argument("--learning_rate", type=float, default=1e-7) # Keep small
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--adap_kl_ctrl", type=bool, default=True) # Keep True
    # --- ✨ Set a reasonable default for init_kl_coef ---
    parser.add_argument("--init_kl_coef", type=float, default=0.2) # Lower starting value
    # ----------------------------------------------------
    parser.add_argument("--target_kl", type=float, default=0.1) # Slightly lower target
    parser.add_argument("--clip_reward", type=float, default=5.0) # Lower clip range post-norm
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_n_steps", type=int, default=0)

    parser.add_argument("--vf_coef", type=float, default=0.1) # Increased slightly
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir, cache_dir=args.cache_dir, padding_side="left")
  
    
    pad_token_id = tokenizer.pad_token_id
    print(pad_token_id)

    special_tokens_dict = {'additional_special_tokens': list(SPECIAL_TOKENS.values())}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} new tokens.")

    end_token = SPECIAL_TOKENS["end"]
    end_token_id = tokenizer.convert_tokens_to_ids(end_token)
    pad_token_id = tokenizer.pad_token_id
    user_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["user"])
    
    print(f"Pad Token ID: {pad_token_id}, End Token ID: {end_token_id}")

    tox_tokenizer = AutoTokenizer.from_pretrained(args.tox_model_dir, cache_dir=args.cache_dir)
    pref_tokenizer = AutoTokenizer.from_pretrained(args.pref_model_dir, cache_dir=args.cache_dir)
    if tox_tokenizer.pad_token is None: tox_tokenizer.pad_token = tox_tokenizer.eos_token
    if pref_tokenizer.pad_token is None: pref_tokenizer.pad_token = pref_tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_model_dir, cache_dir=args.cache_dir)
    
    model.config.pad_token_id = pad_token_id
    model.pretrained_model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = end_token_id
    model.pretrained_model.config.eos_token_id = end_token_id
   

    print("Zero-initializing value head for stability...")
    with torch.no_grad():
        for param in model.v_head.parameters():
            param.zero_()

    model = model.to(device)

    print("Creating reference model...")
    ref_model = create_reference_model(model)

    print("Zero-initializing REF model's value head for stability...")
    with torch.no_grad():
        for param in ref_model.v_head.parameters():
            param.zero_()

    print("resizing both models")
    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    ref_model.pretrained_model.resize_token_embeddings(len(tokenizer))

    ref_model.eval()
    reward_device = torch.device("cpu")
    tox_model = AutoModelForSequenceClassification.from_pretrained(args.tox_model_dir, cache_dir=args.cache_dir).to(reward_device)
    pref_model = AutoModelForSequenceClassification.from_pretrained(args.pref_model_dir, cache_dir=args.cache_dir).to(reward_device)
    tox_model.eval()
    pref_model.eval()

    dataset = get_prompt_dataset(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    print(args.learning_rate)
    ppo_config = PPOConfig(
        model_name=args.sft_model_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        kl_penalty="kl",
        adap_kl_ctrl=args.adap_kl_ctrl,
        init_kl_coef=args.init_kl_coef, 
        target=args.target_kl,
        cliprange=0.1, 
        cliprange_value=0.2,
        log_with="tensorboard",
        accelerator_kwargs={"mixed_precision": "bf16"},
        project_kwargs={"logging_dir": os.path.join(args.output_dir, "logs")},
        use_score_norm=True, # Re-normalize our combined reward
        score_clip=args.clip_reward, # TRL handles clipping after normalization
        seed=args.seed,
        optimize_cuda_cache=True,
        vf_coef=args.vf_coef,           
        max_grad_norm=args.max_grad_norm
    )
    # -----------------------------

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator
    )


    # --- Generation kwargs ---
    generation_kwargs = dict(
        min_length=-1,
        do_sample=True,    
        top_k=50,           
        top_p=0.9,           
        temperature=1.0,
        pad_token_id=pad_token_id,
        eos_token_id=[end_token_id, user_token_id],
        max_new_tokens=args.output_max_len,
    )
    print(generation_kwargs["do_sample"])
    print("Starting PPO training...")
    # Calculate total steps correctly based on epochs
    total_steps_per_epoch = len(ppo_trainer.dataloader)
    total_ppo_steps = total_steps_per_epoch * ppo_config.ppo_epochs
    print(f"Total steps: {total_ppo_steps} ({total_steps_per_epoch} per epoch)")

    global_step = 0

    # --- Explicit Epoch Loop ---
    for epoch in range(ppo_config.ppo_epochs):
        print(f"--- Epoch {epoch+1}/{ppo_config.ppo_epochs} ---")
        for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=total_steps_per_epoch, desc=f"Epoch {epoch+1}"):
            # Data collator provides tensors
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            query_texts = batch.get("query", [""] * input_ids.shape[0]) # Get strings
            bs = input_ids.shape[0]

            # Prepare list of tensors for generate
            query_tensors_list = [input_ids[i].detach() for i in range(bs)]

            start_time = time.time()
            try:
                response_tensors = ppo_trainer.generate(
                    query_tensors_list,
                    return_prompt=False,
                    **generation_kwargs,
                ) 
            except Exception as e:
                print(f"Generation failed at step {step}: {e}")
                gc.collect(); torch.cuda.empty_cache(); continue
            gen_time = time.time() - start_time

            try:
                response_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            except Exception as e:
                print(f"Error decoding responses at step {step}: {e}"); continue

            # --- Compute rewards ---
            reward_start_time = time.time()
            valid_idx = [i for i, r in enumerate(response_texts) if r and r.strip()]
            # Create rewards tensor on the correct device
            rewards_tensor = torch.full((bs,), INVALID_RESPONSE_REWARD, dtype=torch.float32, device=ppo_trainer.accelerator.device)

            if valid_idx:
                valid_prompts = [query_texts[i] for i in valid_idx]
                valid_responses = [response_texts[i] for i in valid_idx]
                combined_scores = get_combined_rewards(
                    valid_prompts, valid_responses,
                    pref_model, pref_tokenizer, tox_model, tox_tokenizer,
                    reward_device, args
                )
                if len(combined_scores) == len(valid_idx):
                    scores_tensor = torch.tensor(combined_scores, dtype=torch.float32, device=rewards_tensor.device)
                    rewards_tensor[torch.tensor(valid_idx, device=rewards_tensor.device)] = scores_tensor
                else:
                    print(f"Warning: Reward/valid mismatch step {step}. Skipping.")

            reward_time = time.time() - reward_start_time

            try:
                rewards_mean = rewards_tensor.mean().item()
                rewards_std = rewards_tensor.std().item()
                rewards_min = rewards_tensor.min().item()
                rewards_max = rewards_tensor.max().item()
                print(f"\n[Step {global_step} Epoch {epoch+1}] Rewards Stats (before TRL norm): "
                    f"Mean={rewards_mean:.4f}, Std={rewards_std:.4f}, "
                    f"Min={rewards_min:.4f}, Max={rewards_max:.4f}")
                # Optional: Log shapes to double-check
                print(f"  Query List Len: {len(query_tensors_list)}, Resp List Len: {len(response_tensors)}")
                print(f"  Rewards Tensor Shape: {rewards_tensor.shape}")
            except Exception as log_e:
                print(f"Error during pre-step logging: {log_e}")

           
            rewards_list_of_tensors = [r for r in rewards_tensor]

            # Perform PPO step
            try:
                stats = ppo_trainer.step(
                    query_tensors_list,  
                    response_tensors,    
                    rewards_list_of_tensors
                )

                vf_loss = stats.get('ppo/loss/value', None)
                policy_loss = stats.get('ppo/loss/policy', None)
                kl_div = stats.get('objective/kl', None)

                # Format safely, printing 'N/A' if value is None
                kl_str = f"{kl_div:.4f}" if kl_div is not None else "N/A"
                vf_loss_str = f"{vf_loss:.4f}" if vf_loss is not None else "N/A"
                policy_loss_str = f"{policy_loss:.4f}" if policy_loss is not None else "N/A"

                print(f"  PPO Step Output: KL={kl_str}, VF Loss={vf_loss_str}, Policy Loss={policy_loss_str}")
                # ----------------------------------------

                stats["custom/gen_time"] = gen_time
                stats["custom/reward_time"] = reward_time
                stats["custom/non_empty_responses"] = len(valid_idx) / bs if bs > 0 else 0.0

                # Log stats using TRL's method 
                ppo_trainer.log_stats(stats, {"query": query_texts[:3]}, rewards_tensor.cpu().tolist())

            except Exception as e:
                print(f"Error during PPO step {step} in epoch {epoch+1}: {e}")
                gc.collect(); torch.cuda.empty_cache(); continue

            global_step += 1
            if global_step % 50 == 0:
                gc.collect(); torch.cuda.empty_cache()

            if args.save_every_n_steps > 0 and (global_step % args.save_every_n_steps == 0):
                print(f"Saving intermediate model at step {global_step}...")
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_dir, exist_ok=True) 
                ppo_trainer.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

    print("PPO Training complete. Saving final model...")
    os.makedirs(args.output_dir, exist_ok=True)
    ppo_trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved to:", args.output_dir)
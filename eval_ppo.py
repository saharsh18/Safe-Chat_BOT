import argparse
import torch
import gc
import random
import numpy as np
import readline # For better interactive input
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

# --- Constants ---
SPECIAL_TOKENS = {
    "user": "<|user|>",
    "bot": "<|bot|>",
    "end": "<|endofbot|>",
}

# --- Helper Functions ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_dialogue_to_messages(dialog_turns):
    """
    (Identical to create_chat_prompt in the training script)
    Formats a list of dialogue utterances into the 'messages' list of dicts format.
    """
    messages = []
    # Assumes even turns are user, odd turns are bot
    for i, utterance in enumerate(dialog_turns):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": utterance})
    return messages

def score_texts(model, tokenizer, texts, device, max_length, batch_size=8):
    """
    Generic function to get raw logits from a reward/classifier model.
    """
    if not texts:
        return []
    
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding="max_length", # Pad to max_length for consistency
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # Get logits. Assumes single-output regression/classification
            logits = model(**inputs).logits.squeeze(-1).cpu().float()
            scores.extend(logits.tolist())
    return scores

# --- Evaluation Functions ---

def qualitative_evaluation(
    ppo_model, ppo_tokenizer, 
    pref_model, pref_tokenizer, 
    tox_model, tox_tokenizer, 
    device, args, 
    num_samples=5
):
    """
    Performs a qualitative evaluation on a small, random sample,
    and scores the generation using the preference and toxicity models.
    """
    print("--- Starting Qualitative Evaluation ---")
    
    try:
        dataset = load_dataset("daily_dialog", split="validation")
        dataset = dataset.shuffle(seed=args.seed).select(range(num_samples))
    except Exception as e:
        print(f"Could not load validation set. Skipping evaluation. Error: {e}")
        return

    # Get the end_token_id from the PPO tokenizer
    end_token_id = ppo_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end"])

    generation_kwargs = dict(
        min_length=-1, top_k=40, top_p=0.9, do_sample=True,
        pad_token_id=end_token_id,
        eos_token_id=end_token_id,
        max_new_tokens=args.output_max_len,
        temperature=0.7, 
    )

    for i, example in enumerate(dataset):
        messages = format_dialogue_to_messages(example["dialog"][:-1])
        ground_truth_response = example["dialog"][-1]
        
        prompt_text = ppo_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = ppo_tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)

        # Generate response
        outputs = ppo_model.generate(**inputs, **generation_kwargs)
        response_text = ppo_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # --- Score the Generation ---
        # 1. Preference Score (Input: Prompt + Response)
        pref_input_text = prompt_text + response_text
        pref_score = score_texts(
            pref_model, pref_tokenizer, [pref_input_text], 
            device, args.pref_max_len
        )[0]

        # 2. Toxicity Score (Input: Response Only)
        tox_score = score_texts(
            tox_model, tox_tokenizer, [response_text], 
            device, args.tox_max_len
        )[0]

        print(f"\n--- Sample {i+1}/{num_samples} ---")
        print(f"Context:\n{prompt_text.replace(ppo_tokenizer.bos_token, '')}")
        print(f"Expected Response:\n{ground_truth_response}")
        print("-" * 20)
        print(f"    PPO-Generated Response:\n{response_text}")
        print("-" * 20)
        print(f"    📈 Preference Score (raw): {pref_score:.4f}")
        print(f"    📉 Toxicity Score (raw):   {tox_score:.4f}")

def interactive_chat(
    ppo_model, ppo_tokenizer, 
    pref_model, pref_tokenizer, 
    tox_model, tox_tokenizer, 
    device, args
):
    """
    Starts an interactive chat session, scoring each response.
    """
    print("\n--- Starting Interactive Chatbot (PPO Model) ---")
    print("Type 'quit' or 'exit' to end the session.")
    
    chat_history = []
    
    # Get the end_token_id from the PPO tokenizer
    end_token_id = ppo_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end"])

    # Use the *exact* generation kwargs from PPO training
    generation_kwargs = dict(
        min_length=-1, top_k=40, top_p=0.9, do_sample=True,
        pad_token_id=end_token_id,
        eos_token_id=end_token_id,
        max_new_tokens=args.output_max_len,
        temperature=0.7, 
    )

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Bot: Goodbye!")
                break
            
            # 1. Format prompt
            chat_history.append({"role": "user", "content": user_input})
            prompt_text = ppo_tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            inputs = ppo_tokenizer(prompt_text, return_tensors="pt").to(device)

            # 2. Generate response
            outputs = ppo_model.generate(**inputs, **generation_kwargs)
            response_text = ppo_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
           
            chat_history.append({"role": "assistant", "content": response_text})

            # Preference Score (Input: Prompt + Response)
            pref_input_text = prompt_text + response_text
            pref_score = score_texts(
                pref_model, pref_tokenizer, [pref_input_text], 
                device, args.pref_max_len
            )[0]

            # Toxicity Score (Input: Response Only)
            tox_score = score_texts(
                tox_model, tox_tokenizer, [response_text], 
                device, args.tox_max_len
            )[0]
            
            # 4. Print results
            print(f"Bot: {response_text}")
            print(f"     [📈 Pref Score: {pref_score:.4f} | 📉 Tox Score: {tox_score:.4f}]")

        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
           

def main():
    parser = argparse.ArgumentParser(description="Evaluate and interact with a PPO-trained model.")
    # --- Model Dirs ---
    parser.add_argument("--ppo_model_dir", type=str, required=True, help="Directory where the PPO model is saved (from ppo-output-stable).")
    parser.add_argument("--tox_model_dir", type=str, required=True, help="Directory of the *trained* toxicity classifier.")
    parser.add_argument("--pref_model_dir", type=str, required=True, help="Directory of the *trained* preference reward model.")
    
    # --- Max Lengths (Must match PPO script) ---
    parser.add_argument("--output_max_len", type=int, default=64, help="Max *new tokens* for generation.")
    parser.add_argument("--tox_max_len", type=int, default=128, help="Max length for toxicity model input.")
    parser.add_argument("--pref_max_len", type=int, default=512, help="Max length for preference model input (prompt + response).")
    
    # --- Other ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 for loading models.")

    # --- Control Flags ---
    parser.add_argument("--run_qualitative", action="store_true", help="Run qualitative evaluation (sample generations).")
    parser.add_argument("--run_chat", action="store_true", help="Start the interactive chatbot.")
    args = parser.parse_args()

    set_seed(args.seed)
    
    run_all = not any([args.run_qualitative, args.run_chat])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32

    # --- 1. Load PPO Model (Generator) ---
    print(f"Loading PPO model from: {args.ppo_model_dir}")
    
    ppo_model = AutoModelForCausalLM.from_pretrained(
        args.ppo_model_dir, 
        torch_dtype=dtype
    ).to(device)
    # Load tokenizer
    ppo_tokenizer = AutoTokenizer.from_pretrained(args.ppo_model_dir, padding_side="left")
    if ppo_tokenizer.pad_token is None:
        ppo_tokenizer.pad_token = ppo_tokenizer.eos_token
    
    # --- 2. Load Preference Model (Reward Model 1) ---
    print(f"Loading Preference model from: {args.pref_model_dir}")
    pref_model = AutoModelForSequenceClassification.from_pretrained(
        args.pref_model_dir, 
        torch_dtype=dtype
    ).to(device)
    pref_tokenizer = AutoTokenizer.from_pretrained(args.pref_model_dir)
    if pref_tokenizer.pad_token is None:
        pref_tokenizer.pad_token = pref_tokenizer.eos_token

    # --- 3. Load Toxicity Model (Reward Model 2) ---
    print(f"Loading Toxicity model from: {args.tox_model_dir}")
    tox_model = AutoModelForSequenceClassification.from_pretrained(
        args.tox_model_dir, 
        torch_dtype=dtype
    ).to(device)
    tox_tokenizer = AutoTokenizer.from_pretrained(args.tox_model_dir)
    if tox_tokenizer.pad_token is None:
        tox_tokenizer.pad_token = tox_tokenizer.eos_token

    # Set all models to evaluation mode
    ppo_model.eval()
    pref_model.eval()
    tox_model.eval()
    
    print("All models loaded successfully.")
    
    # --- Execute requested modes ---
    common_args = (
        ppo_model, ppo_tokenizer,
        pref_model, pref_tokenizer,
        tox_model, tox_tokenizer,
        device, args
    )
    
    if run_all or args.run_chat:
        interactive_chat(*common_args)

    print("Evaluation complete.")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
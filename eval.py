import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import readline 
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

SPECIAL_TOKENS = {
    "user": "<|user|>",
    "bot": "<|bot|>",
    "end": "<|endofbot|>",
}

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

# --- Evaluation Functions ---
def calculate_metrics(model, tokenizer, device, eval_batch_size=2):
    """
    Calculates evaluation loss and perplexity on the validation set
    in a memory-efficient way, using the training script's logic.
    """
    print("Starting Quantitative Evaluation (Loss, Perplexity)...")
    print(f"   - Using batch size: {eval_batch_size}")

    try:
        dataset = load_dataset("daily_dialog", split="validation")
    except Exception as e:
        print(f"Could not load validation set. Skipping metrics. Error: {e}")
        return

    def tokenize_and_prepare_labels(examples):
        prompts = []
        full_texts = []
        for dialog in examples["dialog"]:
            # Use the exact same formatting function as in training
            messages_context = format_dialogue_to_messages(dialog[:-1])
            messages_full = format_dialogue_to_messages(dialog)
            
            prompt_str = tokenizer.apply_chat_template(messages_context, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_str)
            
            full_text_str = tokenizer.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
            full_texts.append(full_text_str)

        model_inputs = tokenizer(full_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        prompt_inputs = tokenizer(prompts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        prompt_lengths = torch.sum(prompt_inputs.attention_mask, dim=1)
        labels = model_inputs.input_ids.clone()
        
        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100
        labels[labels == tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs

    processed_dataset = dataset.map(tokenize_and_prepare_labels, batched=True, remove_columns=dataset.column_names)
    processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataloader = DataLoader(processed_dataset, batch_size=eval_batch_size)

    model.eval()
    total_loss = 0
    num_valid_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Calculating Loss"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            if not math.isnan(loss.item()):
                total_loss += loss.item()
                num_valid_batches += 1

    if num_valid_batches > 0:
        avg_loss = total_loss / num_valid_batches
        perplexity = math.exp(avg_loss)
    else:
        avg_loss, perplexity = float("nan"), float("nan")

    print("\n--- Quantitative Metrics ---")
    print(f"Average Evaluation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}\n")

def qualitative_evaluation(model, tokenizer, device, num_samples=5):
    """
    Performs a qualitative evaluation on a small, random sample,
    using the training script's logic.
    """
    print("Starting Qualitative Evaluation...")

    try:
        dataset = load_dataset("daily_dialog", split="validation")
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    except Exception as e:
        print(f"Could not load validation set. Skipping evaluation. Error: {e}")
        return

    for i, example in enumerate(dataset):
        # Use the exact same formatting function as in training
        messages = format_dialogue_to_messages(example["dialog"][:-1])
        ground_truth_response = example["dialog"][-1]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=True, top_p=0.95, top_k=50, temperature=0.7,
            eos_token_id=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end"])
        )
        model_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print(f"--- Sample {i+1}/{num_samples} ---")
        print(f"Context:\n{prompt.replace(tokenizer.bos_token, '')}")
        print(f"Expected Response:\n{ground_truth_response}")
        print(f"    Generated Response:\n{model_response}\n")

def interactive_chat(model, tokenizer, device):
    """
    Starts an interactive chat session with the fine-tuned model.
    """
    print("Starting Interactive Chatbot")
    print("Type 'quit' or 'exit' to end the session.")
    
    chat_history = []
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Bot: Goodbye!")
                break
            chat_history.append({"role": "user", "content": user_input})
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs, max_new_tokens=150, do_sample=True, top_p=0.95, top_k=50, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["end"])
            )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Bot: {response}")
            chat_history.append({"role": "assistant", "content": response})
        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description="Evaluate and interact with a fine-tuned GPT-2 chatbot.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the model/adapters are saved.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="The name of the base model.")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Batch size for quantitative evaluation.")
    parser.add_argument("--lora", action="store_true", help="LoRA adapters model.")
    
    # Optional flags to control which parts of the script to run
    parser.add_argument("--run_metrics", action="store_true", help="Run quantitative evaluation (loss, perplexity).")
    parser.add_argument("--run_qualitative", action="store_true", help="Run qualitative evaluation (sample generations).")
    parser.add_argument("--run_chat", action="store_true", help="Start the interactive chatbot.")
    args = parser.parse_args()

    # If no specific mode is selected, run all of them by default
    run_all = not any([args.run_metrics, args.run_qualitative, args.run_chat])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Conditional Model Loading ---
    if args.lora:
        print("Loading LoRA model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        print("Resizing token embeddings to match tokenizer...")
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, args.model_dir)
        model = model.merge_and_unload()
    else:
        print("Loading fully fine-tuned model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    model.to(device)
    model.eval()
    
    # --- Execute requested modes ---
    if run_all or args.run_metrics:
        calculate_metrics(model, tokenizer, device, eval_batch_size=args.eval_batch_size)
    
    if run_all or args.run_qualitative:
        qualitative_evaluation(model, tokenizer, device)
    
    if run_all or args.run_chat:
        interactive_chat(model, tokenizer, device)

if __name__ == "__main__":
    main()
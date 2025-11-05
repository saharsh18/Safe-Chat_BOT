import torch
from tqdm import tqdm
import numpy as np
import random

# --- Helper Functions ---

def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_raw_preference_scores(model, tokenizer, texts, batch_size, device, max_length):
    """
    Runs the preference model on a list of texts and returns the raw (logit) scores.
    
    Args:
        model: The loaded preference model.
        tokenizer: The loaded preference model's tokenizer.
        texts: A list of strings (e.g., prompt + response).
        batch_size: How many to process at once.
        device: The torch device (e.g., "cuda").
        max_length: The max sequence length for the tokenizer.
        
    Returns:
        A numpy array of raw scores.
    """
    all_scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(device)
            
            # Get the raw logit (score)
            outputs = model(**inputs)
            scores = outputs.logits.squeeze().cpu() # Shape: (batch_size,)

            if scores.dim() == 0: # Handle batch size of 1
                scores = scores.unsqueeze(0)
                
            all_scores.extend(scores.numpy())
            
    return np.array(all_scores)

def get_toxicity_scores(model, tokenizer, texts, batch_size, device, max_length):
    """
    Runs the toxicity model on a list of texts and returns the safety score (1 - prob_toxic).
    
    Args:
        model: The loaded toxicity model.
        tokenizer: The loaded toxicity model's tokenizer.
        texts: A list of strings (e.g., just the response).
        batch_size: How many to process at once.
        device: The torch device (e.g., "cuda").
        max_length: The max sequence length for the tokenizer.
        
    Returns:
        A numpy array of safety scores (1.0 - toxicity_probability).
    """
    all_scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(device)
            
            # Get the raw logit
            outputs = model(**inputs)
            logits = outputs.logits.squeeze().cpu() 
            
            if logits.dim() == 0: 
                logits = logits.unsqueeze(0)

            # Convert logit to probability and then to safety score
            probs_undesirable = torch.sigmoid(logits)
            safety_scores = 1.0 - probs_undesirable.numpy()
                
            all_scores.extend(safety_scores)
            
    return np.array(all_scores)
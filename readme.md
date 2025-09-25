# 🛡️ Safe Controllable Chatbot (RLHF-style)

This project explores building a **safety-aware chatbot** using **transformer-based models** and **reinforcement learning with human feedback (RLHF)-style fine-tuning**.  
The goal is to align a GPT-2 chatbot with **safety** (reduce toxic replies) and **diversity** (avoid bland repetition), while keeping fluent and useful responses.

---

## ✅ Current Progress

### 1. Fine-tuning GPT-2 (Supervised Fine-Tuning, SFT)
- Base model: **GPT-2** (Hugging Face `openai-community/gpt2`).  
- Dataset: **[DailyDialog](https://huggingface.co/datasets/daily_dialog)** — everyday conversations.  
- Special tokens used for clarity:
  - `<|user|>` – marks the user input  
  - `<|bot|>` – marks the bot reply start  
  - `<|endofbot|>` – marks the end of a bot reply  
- Training setup:
  - Masked user tokens so only the bot’s side contributes to loss.  
  - Used **SFTTrainer** from Hugging Face TRL.  
  - Optionally supports **LoRA adapters** for low-VRAM fine-tuning.  
- Output: `gpt2-sft/` (or `gpt2-sft-lora/`) containing the fine-tuned chatbot.

### 2. Sample Generation
After SFT, the model can already produce more dialogue-like replies:
```text
<|user|> hey, can you help me plan my day?
<|bot|> Sure! Let's start with your morning routine. <|endofbot|>

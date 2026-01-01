import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

from rlhf.utils import sequence_logprob

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("outputs/train_lm")
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Load base (SFT) model
# -------------------------
base_model = AutoModelForCausalLM.from_pretrained("outputs/train_lm").to(device)
base_model.eval()

# -------------------------
# Load GRPO policy (base + LoRA)
# -------------------------
policy_model = AutoModelForCausalLM.from_pretrained("outputs/train_lm")
policy_model = PeftModel.from_pretrained(policy_model, "models/grpo_lora")
policy_model = policy_model.to(device)
policy_model.eval()

# -------------------------
# Helper: generate
# -------------------------
def generate(model, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# GRPO score (group advantage)
# -------------------------
def grpo_advantage(model, chosen_text, rejected_texts):
    # chosen
    chosen = tokenizer(chosen_text, return_tensors="pt", truncation=True).to(device)
    lp_chosen = sequence_logprob(
        model,
        chosen["input_ids"],
        chosen["attention_mask"]
    )

    # rejected group
    rejected_logps = []
    for txt in rejected_texts:
        r = tokenizer(txt, return_tensors="pt", truncation=True).to(device)
        lp_r = sequence_logprob(
            model,
            r["input_ids"],
            r["attention_mask"]
        )
        rejected_logps.append(lp_r)

    rejected_logps = torch.stack(rejected_logps, dim=1)  # [B=1, K]

    all_scores = torch.cat(
        [lp_chosen.unsqueeze(1), rejected_logps],
        dim=1
    )

    advantages = all_scores - all_scores.mean(dim=1, keepdim=True)
    return advantages[:, 0].item()  # chosen advantage

# -------------------------
# Example inference
# -------------------------
prompt = "Explain why reinforcement learning is useful for aligning large language models."

# Generate responses
base_text = generate(base_model, prompt)
policy_text = generate(policy_model, prompt)

print("\n=== PROMPT ===")
print(prompt)

print("\n=== BASE MODEL ===")
print(base_text)

print("\n=== GRPO MODEL ===")
print(policy_text)

# -------------------------
# Group comparison
# -------------------------
rejected_examples = [
    base_text,
    "I don't know.",
    "Reinforcement learning is not important."
]

adv = grpo_advantage(
    policy_model,
    chosen_text=policy_text,
    rejected_texts=rejected_examples
)

print("\n=== GRPO ADVANTAGE ===")
print(f"Chosen advantage over group: {adv:.4f}")

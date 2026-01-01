import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("outputs/train_lm")
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Load base (reference) model
# -------------------------
base_model = AutoModelForCausalLM.from_pretrained("outputs/train_lm").to(device)
base_model.eval()

# -------------------------
# Load DPO policy (base + LoRA)
# -------------------------
policy_model = AutoModelForCausalLM.from_pretrained("outputs/train_lm")
policy_model = PeftModel.from_pretrained(policy_model, "models/dpo_lora")
policy_model = policy_model.to(device)
policy_model.eval()

# -------------------------
# Helper: sequence log-prob
# -------------------------
def sequence_logprob(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1]
        labels = input_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)

        token_logp = token_logp * attention_mask[:, 1:]
        return token_logp.sum(dim=1)  # [B]

# -------------------------
# Generation
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
# Evaluation example
# -------------------------
prompt = "Explain why reinforcement learning is useful for LLM alignment."

base_text = generate(base_model, prompt)
policy_text = generate(policy_model, prompt)

print("\n=== PROMPT ===")
print(prompt)

print("\n=== BASE MODEL ===")
print(base_text)

print("\n=== DPO MODEL ===")
print(policy_text)

# -------------------------
# DPO preference score
# -------------------------
inputs_base = tokenizer(base_text, return_tensors="pt", truncation=True).to(device)
inputs_policy = tokenizer(policy_text, return_tensors="pt", truncation=True).to(device)

logp_base = sequence_logprob(
    base_model,
    inputs_base["input_ids"],
    inputs_base["attention_mask"]
)

logp_policy = sequence_logprob(
    policy_model,
    inputs_policy["input_ids"],
    inputs_policy["attention_mask"]
)

print("\n=== DPO Preference Score ===")
print(f"Policy − Base logp: {(logp_policy - logp_base).item():.4f}")
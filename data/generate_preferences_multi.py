import json
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config
# -------------------------
MODEL_NAME = "gpt2"
NUM_CANDIDATES = 6        # total generations per prompt
NUM_REJECTIONS = 3        # how many rejected to keep
MAX_NEW_TOKENS = 120
TEMPERATURES = [0.7, 0.9, 1.1, 1.3]
TOP_P = 0.95

PROMPTS_FILE = "data/prompts.json"
OUTPUT_FILE = "data/preferences_multi.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# -------------------------
# Load prompts
# -------------------------
with open(PROMPTS_FILE) as f:
    prompts = json.load(f)

# -------------------------
# Generation helper
# -------------------------
@torch.no_grad()
def generate(prompt, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
        top_p=TOP_P
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

# -------------------------
# Heuristic scoring
# -------------------------
def score_response(text):
    """
    Simple but effective:
    - Prefer medium-length answers
    - Penalize very short / very long
    """
    length = len(text.split())
    if length < 10:
        return -10
    if length > 200:
        return 200 - length
    return length

# -------------------------
# Generate multi-rejection preferences
# -------------------------
preference_data = []

for prompt in tqdm(prompts):
    candidates = []

    for i in range(NUM_CANDIDATES):
        temp = random.choice(TEMPERATURES)
        response = generate(prompt, temp)
        score = score_response(response)
        candidates.append((response, score))

    # Sort best → worst
    candidates.sort(key=lambda x: x[1], reverse=True)

    chosen = candidates[0][0]
    rejected = [c[0] for c in candidates[-NUM_REJECTIONS:]]

    preference_data.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })

# -------------------------
# Save
# -------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(preference_data, f, indent=2)

print(f"Saved {len(preference_data)} multi-rejection examples → {OUTPUT_FILE}")

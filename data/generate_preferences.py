import json
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config
# -------------------------
MODEL_NAME = "gpt2"
NUM_SAMPLES = 3        # responses per prompt
MAX_NEW_TOKENS = 100
TEMPERATURES = [0.7, 1.0, 1.3]
OUTPUT_FILE = "data/preferences.json"

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
with open("data/prompts.json") as f:
    prompts = json.load(f)

# -------------------------
# Generation helper
# -------------------------
@torch.no_grad()
def generate(prompt, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
        top_p=0.95
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

# -------------------------
# Simple heuristic scoring
# -------------------------
def heuristic_score(text):
    """
    Simple automatic preference:
    - longer is better (up to a point)
    - penalize very short / empty
    """
    length = len(text.split())
    return min(length, 200)

# -------------------------
# Generate preferences
# -------------------------
preference_data = []

for prompt in tqdm(prompts):
    candidates = []

    for t in TEMPERATURES[:NUM_SAMPLES]:
        out = generate(prompt, t)
        score = heuristic_score(out)
        candidates.append((out, score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    chosen = candidates[0][0]
    rejected = candidates[-1][0]

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

print(f"Saved {len(preference_data)} preference pairs to {OUTPUT_FILE}")

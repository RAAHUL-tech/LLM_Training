import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Load Inference Config
# -------------------------
with open("configs/inference.yaml") as f:
    cfg = yaml.safe_load(f)

# -------------------------
# Device
# -------------------------
if cfg["device"] == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = cfg["device"]

# -------------------------
# Tokenizer & Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"])

# Pad token handling
if cfg["pad_token"] == "eos":
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"])

if cfg.get("use_fp16", False) and device == "cuda":
    model = model.half()

model.to(device)
model.eval()

# -------------------------
# Generation Function
# -------------------------
@torch.no_grad()
def generate(prompt: str):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg["max_input_length"],
    ).to(device)

    generation_kwargs = {
        "max_new_tokens": cfg["max_new_tokens"],
        "do_sample": cfg["do_sample"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": cfg["repetition_penalty"],
    }

    # Optional top-k
    if cfg.get("top_k") is not None:
        generation_kwargs["top_k"] = cfg["top_k"]

    outputs = model.generate(**inputs, **generation_kwargs)

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=cfg["skip_special_tokens"]
    )

# -------------------------
# Example Run
# -------------------------
if __name__ == "__main__":
    prompt = "Once upon a time in a futuristic city,"
    text = generate(prompt)
    print(text)
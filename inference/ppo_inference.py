import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rlhf.reward_model import RewardModel
from rlhf.utils import sequence_logprob

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("outputs/train_lm")
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained("outputs/train_lm").to(device)
policy = PeftModel.from_pretrained(base, "models/ppo_lora").to(device)
policy.eval()

@torch.no_grad()
def generate(model, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "Explain reinforcement learning from human feedback."

print("=== Base Model ===")
print(generate(base, prompt))

print("\n=== PPO-RLHF Model ===")
print(generate(policy, prompt))

reward_model = RewardModel("outputs/train_lm").to(device)
reward_model.reward_head.load_state_dict(
    torch.load("models/reward_model/reward_head.pt", map_location=device)
)
reward_model.eval()

@torch.no_grad()
def score(text):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    return reward_model(
        tokens["input_ids"],
        tokens["attention_mask"]
    ).item()

base_out = generate(base, prompt)
ppo_out = generate(policy, prompt)

print("Reward(Base):", score(base_out))
print("Reward(PPO): ", score(ppo_out))


@torch.no_grad()
def kl_div(prompt):
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    logp_policy = sequence_logprob(
        policy, tokens["input_ids"], tokens["attention_mask"]
    )
    logp_ref = sequence_logprob(
        base, tokens["input_ids"], tokens["attention_mask"]
    )
    return (logp_policy - logp_ref).item()

print("KL divergence:", kl_div(prompt))

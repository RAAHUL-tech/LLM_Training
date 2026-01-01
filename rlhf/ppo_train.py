import yaml
import torch
import wandb
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlhf.reward_model import RewardModel
from rlhf.dataset import PreferenceDataset
from rlhf.utils import sequence_logprob

# -------------------------
# Config
# -------------------------
with open("configs/train_base.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda"
wandb.init(project="llm-from-scratch", name="PPO-LoRA")

# -------------------------
# Models
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("outputs/train_lm")
tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained("outputs/train_lm")

ref = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(device)
ref.eval()
for p in ref.parameters():
    p.requires_grad = False

reward_model = RewardModel("outputs/train_lm").to(device)
reward_model.reward_head.load_state_dict(
    torch.load("models/reward_model/reward_head.pt", map_location=device)
)
reward_model.eval()

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

policy = get_peft_model(policy, lora_cfg).to(device)
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)


# -------------------------
# Data
# -------------------------
dataset = load_dataset("json", data_files="data/preferences.json")["train"]
loader = DataLoader(
    PreferenceDataset(dataset, tokenizer, cfg["max_length"]),
    batch_size=cfg["batch_size"],
    shuffle=True
)

# -------------------------
# PPO Training.        L=min(rt ​At​,clip(rt​,1−ϵ,1+ϵ) At​)− βKL(π∣∣πref​)
# -------------------------
clip_eps = 0.2
kl_beta = 0.1

for epoch in range(3):
    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        logp_new = sequence_logprob(
            policy,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"]
        )

        with torch.no_grad():
            logp_old = sequence_logprob(
                ref,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"]
            )
            reward = reward_model(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"]
            ).squeeze()

        ratio = torch.exp(logp_new - logp_old)
        advantage = reward - reward.mean()

        unclipped = ratio * advantage
        clipped = torch.clamp(
            ratio, 1 - clip_eps, 1 + clip_eps
        ) * advantage

        policy_loss = -torch.min(unclipped, clipped).mean()
        kl = (logp_new - logp_old).mean()

        loss = policy_loss + kl_beta * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({
            "ppo/loss": loss.item(),
            "ppo/reward": reward.mean().item(),
            "ppo/kl": kl.item()
        })

policy.save_pretrained("models/ppo_lora")
wandb.finish()

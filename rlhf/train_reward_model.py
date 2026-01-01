import yaml
import torch
import wandb
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from rlhf.reward_model import RewardModel
from rlhf.dataset import PreferenceDataset

# -------------------------
# Config
# -------------------------
with open("configs/train_base.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda")
wandb.init(project="llm-from-scratch", name="reward-model")

# -------------------------
# Data
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
dataset = load_dataset("json", data_files="data/preferences.json")["train"]

pref_dataset = PreferenceDataset(
    dataset,
    tokenizer,
    cfg["max_length"]
)

loader = DataLoader(pref_dataset, batch_size=cfg["batch_size"], shuffle=True)

# -------------------------
# Model
# -------------------------
model = RewardModel(cfg["model_name"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# -------------------------
# Pairwise Ranking Loss
# -------------------------
def reward_loss(r_chosen, r_rejected):
    return -F.logsigmoid(r_chosen - r_rejected).mean()

# -------------------------
# Training Loop
# -------------------------
model.train()
for epoch in range(3):
    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        r_chosen = model(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"]
        )
        r_rejected = model(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"]
        )

        loss = reward_loss(r_chosen, r_rejected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"reward_loss": loss.item()})

model.save_pretrained("models/reward_model")
wandb.finish()

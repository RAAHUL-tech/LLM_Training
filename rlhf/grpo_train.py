import yaml
import torch
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlhf.dataset import MultiPreferenceDataset
from rlhf.utils import sequence_logprob

# -------------------------
# Config
# -------------------------
with open("configs/train_base.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda"
wandb.init(project="llm-from-scratch", name="GRPO-LoRA-Multi")

# -------------------------
# Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
policy = get_peft_model(
    policy,
    LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM
    )
).to(device)

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)

# -------------------------
# Data
# -------------------------
dataset = load_dataset(
    "json",
    data_files="data/preferences_multi.json"
)["train"]

loader = DataLoader(
    MultiPreferenceDataset(dataset, tokenizer, cfg["max_length"]),
    batch_size=cfg["batch_size"],
    shuffle=True
)

# -------------------------
# GRPO Training (TRUE GROUP VERSION)
# -------------------------
policy.train()

for epoch in range(3):
    for batch in tqdm(loader):

        # ----- chosen -----
        chosen_ids = batch["chosen"]["input_ids"].to(device)
        chosen_mask = batch["chosen"]["attention_mask"].to(device)

        lp_chosen = sequence_logprob(
            policy,
            chosen_ids,
            chosen_mask
        )

        # ----- rejected group -----
        rejected_logps = []

        for r in batch["rejected"]:
            r_ids = r["input_ids"].to(device)
            r_mask = r["attention_mask"].to(device)

            lp_r = sequence_logprob(
                policy,
                r_ids,
                r_mask
            )
            rejected_logps.append(lp_r)

        # Shape: [B, K]
        rejected_logps = torch.stack(rejected_logps, dim=1)

        # ----- group normalization -----
        all_scores = torch.cat(
            [lp_chosen.unsqueeze(1), rejected_logps],
            dim=1
        )  # [B, 1 + K]

        advantages = all_scores - all_scores.mean(dim=1, keepdim=True)

        # ----- GRPO loss -----
        loss = -advantages[:, 0].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({
            "grpo/loss": loss.item(),
            "grpo/advantage_chosen": advantages[:, 0].mean().item(),
            "grpo/num_rejected": rejected_logps.size(1)
        })

policy.save_pretrained("models/grpo_lora")
wandb.finish()

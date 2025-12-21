import math
import yaml
import torch
import wandb

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from datasets import load_dataset
from tqdm import tqdm

# -------------------------
# Load config
# -------------------------
with open("configs/train_base.yaml") as f:
    cfg = yaml.safe_load(f)

torch.manual_seed(cfg["seed"])
device = torch.device("cuda")

# -------------------------
# Initialize W&B
# -------------------------
wandb.init(
    project="llm-from-scratch",
    config=cfg,
    name="train_amp_ckpt"
)

# -------------------------
# Tokenizer & Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
model.gradient_checkpointing_enable()  # 🔥 Gradient checkpointing
model.to(device)

# -------------------------
# Dataset
# -------------------------
dataset = load_dataset(
    cfg["dataset_name"],
    cfg["dataset_config"]
)

def tokenize_fn(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=cfg["max_length"],
        padding=False,
    )
    input_ids = [ids for ids in tokens["input_ids"] if len(ids) > 0]
    return {"input_ids": input_ids}

tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

train_loader = DataLoader(
    tokenized["train"],
    batch_size=cfg["batch_size"],
    shuffle=True,
    collate_fn=data_collator
)

# -------------------------
# Optimizer & Scheduler
# -------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(cfg["learning_rate"]),
    weight_decay=float(cfg["weight_decay"])
)

num_training_steps = cfg["num_epochs"] * len(train_loader)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

scaler = GradScaler()  # 🔥 AMP scaler

# -------------------------
# Training Loop
# -------------------------
def sanity_check(batch):
    assert batch["input_ids"].dim() == 2
    assert batch["input_ids"].size(1) > 0

model.train()
global_step = 0

for epoch in range(cfg["num_epochs"]):
    for batch in tqdm(train_loader):
        sanity_check(batch)
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast():  # 🔥 AMP
            outputs = model(**batch)
            loss = outputs.loss

        # Backprop with AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()

        # Log metrics to W&B
        wandb.log({
            "train/loss": loss.item(),
            "train/perplexity": math.exp(loss.item()),
            "train/lr": lr_scheduler.get_last_lr()[0],
            "train/global_step": global_step
        })
        global_step += 1

    # Epoch-level logging
    print(f"Epoch {epoch} Loss: {loss.item():.4f} | Perplexity: {math.exp(loss.item()):.2f}")
    wandb.log({
        "epoch/loss": loss.item(),
        "epoch/perplexity": math.exp(loss.item()),
        "epoch": epoch
    })

# -------------------------
# Save
# -------------------------
model.save_pretrained(cfg["output_dir"])
tokenizer.save_pretrained(cfg["output_dir"])

wandb.finish()
torch.cuda.max_memory_allocated()
import os
import math
import yaml
import torch
import torch.distributed as dist
import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from datasets import load_dataset
from tqdm import tqdm

# -------------------------
# DDP setup
# -------------------------
def setup_ddp():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

rank, world_size = setup_ddp()

# -------------------------
# Load config
# -------------------------
with open("configs/train_base.yaml") as f:
    cfg = yaml.safe_load(f)

torch.manual_seed(cfg["seed"])

device = torch.device("cpu")  # CPU-safe DDP

# -------------------------
# W&B init (ONLY rank 0)
# -------------------------
if rank == 0:
    wandb.init(
        project="llm-from-scratch",
        name=cfg.get("wandb_run_name", "ddp-cpu-debug"),
        config=cfg
    )

# -------------------------
# Tokenizer & Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
model.to(device)

model = DDP(model)

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

sampler = DistributedSampler(
    tokenized["train"],
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

train_loader = DataLoader(
    tokenized["train"],
    batch_size=cfg["batch_size"],
    sampler=sampler,
    collate_fn=data_collator
)

# -------------------------
# Optimizer & Scheduler
# -------------------------
cfg["learning_rate"] = float(cfg["learning_rate"])
cfg["weight_decay"] = float(cfg["weight_decay"])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["learning_rate"],
    weight_decay=cfg["weight_decay"]
)

num_training_steps = cfg["num_epochs"] * len(train_loader)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# -------------------------
# Training Loop
# -------------------------
def sanity_check(batch):
    assert batch["input_ids"].dim() == 2
    assert batch["input_ids"].size(1) > 0

model.train()
global_step = 0

for epoch in range(cfg["num_epochs"]):
    sampler.set_epoch(epoch)

    for batch in tqdm(train_loader, disable=(rank != 0)):
        sanity_check(batch)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if rank == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": global_step
            })

        global_step += 1

    if rank == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# -------------------------
# Save only on rank 0
# -------------------------
if rank == 0:
    model.module.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    wandb.finish()

cleanup_ddp()
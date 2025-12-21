import time
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

# PEFT LoRA
from peft import LoraConfig, get_peft_model, TaskType

# -------------------------
# Load config
# -------------------------
with open("configs/train_base.yaml") as f:
    cfg = yaml.safe_load(f)

torch.manual_seed(cfg["seed"])
device = torch.device("cuda")

# -------------------------
# W&B init
# -------------------------
wandb.init(project="llm-from-scratch", config=cfg, name="LoRA-Training")

# -------------------------
# Tokenizer & Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
model.gradient_checkpointing_enable()  # gradient checkpointing
model.to(device)

# -------------------------
# Freeze base model & Add LoRA
# -------------------------
for param in model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # check trainable params

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
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

scaler = GradScaler()  # AMP

# -------------------------
# Training Loop + W&B logging
# -------------------------
def sanity_check(batch):
    assert batch["input_ids"].dim() == 2
    assert batch["input_ids"].size(1) > 0
    
model.train()
global_step = 0
step_times = []

for epoch in range(cfg["num_epochs"]):
    for batch in tqdm(train_loader):
        sanity_check(batch)
        batch = {k: v.to(device) for k, v in batch.items()}

        start = time.time()
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()
        end = time.time()

        step_time = end - start
        step_times.append(step_time)

        # Log GPU memory
        max_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        lr = lr_scheduler.get_last_lr()[0]

        # W&B logging
        wandb.log({
            "train/loss": loss.item(),
            "train/perplexity": math.exp(loss.item()),
            "train/lr": lr,
            "train/max_memory_MB": max_mem,
            "train/step_time_s": step_time,
            "train/global_step": global_step
        })

        global_step += 1

        tqdm.write(
            f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | "
            f"Mem: {max_mem:.1f}MB | Step Time: {step_time:.2f}s"
        )

# -------------------------
# Save LoRA adapters only
# -------------------------
model.save_pretrained(cfg["output_dir"] + "_lora")
tokenizer.save_pretrained(cfg["output_dir"])

wandb.finish()
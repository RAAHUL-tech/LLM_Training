import math
import yaml
import torch
from torch.utils.data import DataLoader
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

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Tokenizer & Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
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

    # FILTER EMPTY SEQUENCES
    input_ids = []
    for ids in tokens["input_ids"]:
        if len(ids) > 0:
            input_ids.append(ids)

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
# Optimizer
# -------------------------
# ---- Defensive casting ----
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

def sanity_check(batch):
    assert batch["input_ids"].dim() == 2
    assert batch["input_ids"].size(1) > 0
    
# -------------------------
# Training Loop
# -------------------------
model.train()
progress = tqdm(range(num_training_steps))

for epoch in range(cfg["num_epochs"]):
    for batch in train_loader:
        sanity_check(batch)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress.update(1)
        progress.set_postfix(loss=loss.item())

# -------------------------
# Perplexity
# -------------------------
ppl = math.exp(loss.item())
print(f"Final Perplexity: {ppl:.2f}")

model.save_pretrained(cfg["output_dir"])
tokenizer.save_pretrained(cfg["output_dir"])
import yaml
import torch
import wandb
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlhf.dataset import PreferenceDataset

# -------------------------
# Config
# -------------------------
with open("configs/train_base.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda")
wandb.init(project="llm-from-scratch", name="DPO-LoRA")

# -------------------------
# Tokenizer & Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
base_model.to(device)
base_model.eval()  # reference model

policy_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM
)

policy_model = get_peft_model(policy_model, lora_cfg).to(device)

# -------------------------
# Data
# -------------------------
dataset = load_dataset("json", data_files="data/preferences.json")["train"]
pref_dataset = PreferenceDataset(dataset, tokenizer, cfg["max_length"])
loader = DataLoader(pref_dataset, batch_size=cfg["batch_size"], shuffle=True)

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

# -------------------------
# DPO Loss.      logσ(β[(πc​−πr​)−(πref,c​− πref,r​)])
# -------------------------
def dpo_loss(policy_chosen, policy_rejected,
             ref_chosen, ref_rejected,
             beta=0.1):
    pi_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

# -------------------------
# Training Loop
# -------------------------
policy_model.train()
with torch.no_grad():
    base_model.eval()

for epoch in range(3):
    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        def logp(model, ids, mask):
            out = model(input_ids=ids, attention_mask=mask)
            return out.logits[:, -1, :].log_softmax(-1).mean()

        p_c = logp(policy_model,
                   batch["chosen_input_ids"],
                   batch["chosen_attention_mask"])
        p_r = logp(policy_model,
                   batch["rejected_input_ids"],
                   batch["rejected_attention_mask"])

        with torch.no_grad():
            r_c = logp(base_model,
                       batch["chosen_input_ids"],
                       batch["chosen_attention_mask"])
            r_r = logp(base_model,
                       batch["rejected_input_ids"],
                       batch["rejected_attention_mask"])

        loss = dpo_loss(p_c, p_r, r_c, r_r)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"dpo_loss": loss.item()})

policy_model.save_pretrained("models/dpo_lora")
wandb.finish()

# LLM Training From Scratch — Systems + RLHF

End-to-end implementation of **LLM pretraining, fine-tuning, and RLHF** with a strong focus on **systems correctness**, **memory efficiency**, and **modern preference optimization methods (DPO / GRPO / PPO)**.

This repository is designed to be:
- ✅ **Single-GPU friendly** (Kaggle / Colab)
- ✅ **Distributed-correct** (CPU DDP simulation)
- ✅ **Industry-aligned** (LoRA, AMP, GRPO, logging, inference optimization)

---

## 📌 What This Project Covers

### 1️⃣ Language Model Training
- Causal LM training from HuggingFace models
- AMP (mixed precision)
- Gradient checkpointing
- Distributed Data Parallel (DDP)
- LoRA fine-tuning

### 2️⃣ RLHF & Preference Optimization
- Preference dataset generation
- Reward modeling
- PPO (policy gradient RLHF)
- DPO (Direct Preference Optimization)
- GRPO (Group-based preference optimization, multi-rejection)

### 3️⃣ Inference & Deployment
- Batched inference
- LoRA loading
- DPO / PPO / GRPO inference scripts
- CPU & GPU compatible

### 4️⃣ Experiment Tracking
- Weights & Biases logging
- Loss, reward, KL, advantage tracking

---

## 📂 Project Structure

```

llm-from-scratch/
├── configs/
│   ├── train_base.yaml          # Training hyperparameters
│   └── inference.yaml           # Inference settings
│
├── data/
│   ├── prompts.json             # Base prompts
│   ├── generate_preferences.py  # Pairwise preference generation
│   └── generate_preferences_multi.py  # Multi-rejection (GRPO)
│
├── train/
│   ├── train_lm.py               # Baseline LM training
│   ├── train_lm_ddp.py           # DDP training (CPU/GPU)
│   ├── train_lm_amp_ckpt.py      # AMP + checkpointing
│   ├── train_lm_lora.py          # LoRA fine-tuning
│   └── train_lm_wandb.py         # Training with W&B logging
│
├── rlhf/
│   ├── dataset.py                # Preference datasets
│   ├── reward_model.py           # Reward model definition
│   ├── train_reward_model.py     # Reward model training
│   ├── dpo_train.py              # DPO training
│   ├── ppo_train.py              # PPO training
│   ├── grpo_train.py             # GRPO training (multi-rejection)
│   └── utils.py                  # Logprob utilities
│
├── inference/
│   ├── inference.py              # Base inference
│   ├── dpo_inference.py          # DPO inference
│   ├── ppo_inference.py          # PPO inference
│   └── grpo_inference.py         # GRPO inference
│
├── notebooks/
│   └── llm-training.ipynb        # Interactive experimentation
│
├── models/                       # Saved checkpoints
├── report/                       # Final report & results
├── README.md
├── LICENSE
└── .gitignore

````

---

## Getting Started

### 1️⃣ Environment Setup

```bash
pip install -r requirements.txt
````

Minimum dependencies:

* `torch`
* `transformers`
* `datasets`
* `peft`
* `wandb`
* `accelerate`

---

## ⚙️ Configuration

### `configs/train_base.yaml`

Controls:

* model name
* batch size
* max sequence length
* learning rate
* epochs

Example:

```yaml
model_name: gpt2
batch_size: 4
max_length: 512
lr: 1e-5
epochs: 3
```

---

## Training

### 🔹 Baseline LM Training

```bash
python train/train_lm.py
```

### 🔹 DDP (CPU/GPU Safe)

```bash
torchrun --nproc_per_node=2 train/train_lm_ddp.py
```

### 🔹 AMP + Checkpointing

```bash
python train/train_lm_amp_ckpt.py
```

### 🔹 LoRA Fine-Tuning

```bash
python train/train_lm_lora.py
```

---

## RLHF Pipeline

### 1️⃣ Generate Preference Data

#### Pairwise (DPO / PPO)

```bash
python data/generate_preferences.py
```

#### Multi-Rejection (GRPO)

```bash
python data/generate_preferences_multi.py
```

---

### 2️⃣ Train Reward Model

```bash
python rlhf/train_reward_model.py
```

---

### 3️⃣ Preference Optimization

#### 🔹 DPO

```bash
python rlhf/dpo_train.py
```

#### 🔹 PPO

```bash
python rlhf/ppo_train.py
```

#### 🔹 GRPO (Multi-Rejection)

```bash
python rlhf/grpo_train.py
```

✔ No reward model required
✔ Group-normalized advantages
✔ Lower variance than PPO

---

## 🧪 Inference

### Base

```bash
python inference/inference.py
```

### DPO / PPO / GRPO

```bash
python inference/dpo_inference.py
python inference/ppo_inference.py
python inference/grpo_inference.py
```

Supports:

* LoRA adapters
* Batched decoding
* CPU & GPU

---

## 📊 Experiment Tracking (W&B)

All training scripts support:

* Loss curves
* Reward / advantage
* KL divergence (PPO)
* Throughput

```bash
wandb login
```

Runs are logged under:

```
project = "llm-from-scratch"
```

---

## 📌 Future Extensions

* FlashAttention
* Quantized inference (4-bit)
* Tensor parallelism (Megatron-style)
* Safety & bias evaluation
* Model evaluation (MAUVE / BERTScore)

---

## 📜 License

MIT License — free to use, modify, and learn from.

---
```
```

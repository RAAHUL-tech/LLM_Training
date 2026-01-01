import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, -1]
        reward = self.reward_head(pooled)
        return reward

    def save_pretrained(self, save_directory):
        self.backbone.save_pretrained(save_directory)
        torch.save(
            self.reward_head.state_dict(),
            f"{save_directory}/reward_head.pt"
        )

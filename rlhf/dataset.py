from torch.utils.data import Dataset
"""Dataset for preference-based learning from human feedback.
Preference Dataset Format:
{
  "prompt": "Explain transformers",
  "chosen": "Transformers use attention...",
  "rejected": "Transformers are like RNNs..."
}
"""
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        chosen = self.encode(item["prompt"] + item["chosen"])
        rejected = self.encode(item["prompt"] + item["rejected"])

        return {
            "chosen_input_ids": chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
        }

class MultiPreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        chosen = self.encode(item["prompt"] + item["chosen"])

        rejected = [
            self.encode(item["prompt"] + r)
            for r in item["rejected"]
        ]

        return {
            "chosen": {
                "input_ids": chosen["input_ids"].squeeze(0),
                "attention_mask": chosen["attention_mask"].squeeze(0),
            },
            "rejected": [
                {
                    "input_ids": r["input_ids"].squeeze(0),
                    "attention_mask": r["attention_mask"].squeeze(0),
                }
                for r in rejected
            ]
        }

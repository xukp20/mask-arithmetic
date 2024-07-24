"""
    Load dataset and create dataloader from the dataset of jsonl.
    Like {
        "input_ids": list of int,
        "attention_mask": list of int, 0 or 1
        "labels": list of int
    }
"""

import torch

from torch.utils.data import DataLoader, Dataset

import json

class JsonlDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # map to torch tensor
        return {
            "input_ids": torch.tensor(self.data[idx]["input_ids"]),
            "attention_mask": torch.tensor(self.data[idx]["attention_mask"]),
            "labels": torch.tensor(self.data[idx]["labels"])
        }



# data collator for huggingface transformers training
def default_data_collator(examples):
    result = dict()
    for key in examples[0].keys():
        result[key] = torch.stack([i[key] for i in examples])
    return result


# dataloader for custom training
def create_dataloader(file_path, batch_size=256, shuffle=True, num_workers=0):
    dataset = JsonlDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
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
    def __init__(self, path, sizes=None):
        self.data = []
        if isinstance(path, str):
            with open(path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
            if sizes:
                self.data = self.data[:sizes]
            print(f"Loaded {len(self.data)} data from {path}")
        else:
            for i, file in enumerate(path):
                file_data = []
                with open(file, 'r') as f:
                    for line in f:
                        file_data.append(json.loads(line))
                if sizes:
                    file_data = file_data[:sizes[i]]
                    if sizes[i] > len(file_data):
                        print(f"Warning: file {file} has less data than specified size {sizes[i]}, Dulicating data")
                        file_data = file_data * (sizes[i] // len(file_data)) + file_data[:sizes[i] % len(file_data)]
                self.data.extend(file_data)
                print(f"Loaded {len(file_data)} data from {file}")
                
            import random
            random.shuffle(self.data)
    
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
def create_dataloader(file_path, batch_size=256, shuffle=True, num_workers=0, sizes=None):
    dataset = JsonlDataset(file_path, sizes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
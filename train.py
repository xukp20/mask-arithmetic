import json
import torch
from torch.utils.data import Dataset, DataLoader

# NOTE: change the suffix for different task:
# 1. ans for only answer mask
# 2. num for only operand mask
# 3. mix for both answer and operand mask
# 0716: add def, eq tasks
import os
BASE_DATA_PATH = "./data"
TRAIN_PATHS = [
    "train_def_eq.jsonl"
]

TRAIN_PATHS = [os.path.join(BASE_DATA_PATH, path) for path in TRAIN_PATHS]

EVAL_PATHS = [
    "eval_def_def.jsonl",
    "eval_def_num.jsonl",
    "eval_eq.jsonl",
]

EVAL_PATHS = [os.path.join(BASE_DATA_PATH, path) for path in EVAL_PATHS]

print(TRAIN_PATHS)
print(EVAL_PATHS)

class JSONLDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        if isinstance(file_path, list):
            for path in file_path:
                with open(path, 'r') as f:
                    for line in f:
                        self.data.append(json.loads(line))
        else:
            with open(file_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }

def create_dataloader(file_path, batch_size=256, shuffle=True, num_workers=0):
    dataset = JSONLDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

import os
train_dataloader = create_dataloader(TRAIN_PATHS, batch_size=1024, shuffle=True)
val_dataloaders = {
    os.path.basename(eval_path): create_dataloader(eval_path, batch_size=1024, shuffle=False)
    for eval_path in EVAL_PATHS
}

# prepare the model
from modeling import create_custom_bert_model
from modeling_def import create_custom_def_model

from tokenizer import DEFAULT_TOKENIZER
DEFAULT_CONFIG={
    "vocab_size": DEFAULT_TOKENIZER.get_vocab_size(),
    "hidden_size": 128,
    "num_hidden_layers": 16,
    "num_attention_heads": 8,
    "intermediate_size": 128 * 4,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 16,
    "pad_token_id": DEFAULT_TOKENIZER.pad_token_id,
}

# model = create_custom_bert_model(**DEFAULT_CONFIG)
model = create_custom_def_model()

# train
from tqdm import tqdm
from torch.optim import AdamW

def train_mlm(model, train_loader, val_loaders, epochs=3, learning_rate=2e-5, device="cpu"):
    # check the number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            tqdm.write(f'Loss: {loss.item():.4f}')

        avg_train_loss = total_loss / len(train_loader)
        print(f'Average train loss: {avg_train_loss:.4f}')

        # Validation
        model.eval()
        
        with torch.no_grad():
            for val_name, val_loader in val_loaders.items():
                total_eval_loss = 0
                for batch in tqdm(val_loader, desc='Validation'):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_eval_loss += loss.item()

                avg_val_loss = total_eval_loss / len(val_loader)
                print(f'Average validation loss on {val_name}: {avg_val_loss:.4f}')


    return model


if __name__ == "__main__":
    model = train_mlm(model, train_dataloader, val_dataloaders, epochs=25, learning_rate=2e-4, device="cuda:0")
    model.save_pretrained("./def_model")
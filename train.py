import json
import torch
from torch.utils.data import Dataset, DataLoader

TRAIN_PATH = "./data/train_ans.jsonl"
EVAL_PATH = "./data/eval_ans.jsonl"

class JSONLDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
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

train_dataloader = create_dataloader(TRAIN_PATH, batch_size=256, shuffle=True)
val_dataloader = create_dataloader(EVAL_PATH, batch_size=256, shuffle=False)


# prepare the model
from modeling import create_custom_bert_model
from tokenizer import DEFAULT_TOKENIZER
DEFAULT_CONFIG={
    "vocab_size": DEFAULT_TOKENIZER.get_vocab_size(),
    "hidden_size": 128,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "intermediate_size": 128 * 4,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 16,
    "pad_token_id": DEFAULT_TOKENIZER.pad_token_id,
}

model = create_custom_bert_model(**DEFAULT_CONFIG)

# train
from tqdm import tqdm
from torch.optim import AdamW

def train_mlm(model, train_loader, val_loader, epochs=3, learning_rate=2e-5, device="cpu"):
    # check the number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    # cosine
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
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
        total_eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(val_loader)
        print(f'Average validation loss: {avg_val_loss:.4f}')

    return model


if __name__ == "__main__":
    model = train_mlm(model, train_dataloader, val_dataloader, epochs=15, learning_rate=1e-3, device="cuda:0")
    model.save_pretrained("./model")
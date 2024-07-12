MODEL_PATH="./model"

# Load the model
from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained(MODEL_PATH)

# load eval data
from train import JSONLDataset
from tokenizer import DEFAULT_TOKENIZER

# evaluate
# NOTE: tested on all three eval datasets
for eval_path in ["./data/eval_ans.jsonl", "./data/eval_num.jsonl", "./data/eval_mix.jsonl"]:
    eval_dataset = JSONLDataset(eval_path)

    correct = 0
    total = 0
    for sample in eval_dataset:
        input_ids = sample['input_ids'].unsqueeze(0)
        attention_mask = sample['attention_mask'].unsqueeze(0)
        labels = sample['labels'].unsqueeze(0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predicted = logits.argmax(-1)
        
        # look for the mask token in the input
        mask_id = DEFAULT_TOKENIZER.get_mask_token_id()
        mask_idx = (input_ids == mask_id).nonzero(as_tuple=True)
        mask_idx = mask_idx[1]

        # compare the predicted token with the label
        predicted = predicted[0][mask_idx]
        labels = labels[0][mask_idx]

        total += 1
        if predicted == labels:
            correct += 1
        

    print(f"Accuracy for {eval_path}: {correct / total * 100:.2f}%")